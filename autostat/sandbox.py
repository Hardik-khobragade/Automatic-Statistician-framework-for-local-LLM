"""LAYER 2 (part 1): Sandboxed code execution.

Design goals:
  1. Statefulness -- variables/dataframes defined in one execute_python call
     must be visible in the next call (like a notebook kernel), since the
     agent builds up an analysis over several ReAct turns.
  2. Crash/hang isolation -- LLM-generated code can be buggy (infinite
     loops, huge allocations); each call runs in its own subprocess with a
     wall-clock timeout, so a bad step can't take down the whole agent run.
  3. A best-effort safety net -- an AST whitelist blocks the most obviously
     dangerous operations (filesystem/network/process access, dynamic
     `eval`/`exec`/`__import__`, dunder-attribute probing).

Honest caveat (read this before trusting this in a multi-user setting):
this is *not* a hardened security boundary against adversarial code. It
is meant to catch accidents and obviously-bad LLM output for a single
trusted local user running their own model on their own machine. If you
need to execute genuinely untrusted code, run this whole process inside a
container/VM with no filesystem/network access, or swap in a real sandbox
(gVisor, firecracker-based runners, Docker with seccomp, etc).
"""
from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cloudpickle

# --------------------------------------------------------------------------- #
# Static safety check
# --------------------------------------------------------------------------- #

ALLOWED_TOP_LEVEL_MODULES = {
    "pandas", "numpy", "scipy", "statsmodels", "pingouin",
    "matplotlib", "seaborn", "math", "statistics", "itertools",
    "collections", "re", "json", "warnings", "stats_toolkit",
}

BLOCKED_CALL_NAMES = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "exit", "quit", "globals", "locals", "vars", "breakpoint",
    "getattr", "setattr", "delattr",
}

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib",
    "ctypes", "multiprocessing", "threading", "asyncio", "requests",
    "urllib", "http", "ftplib", "telnetlib", "pickle",
}


class UnsafeCodeError(Exception):
    pass


def check_code_safety(code: str) -> None:
    """Raise UnsafeCodeError if the code uses obviously dangerous
    constructs. Best-effort static check, see module docstring."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeCodeError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_MODULES:
                    raise UnsafeCodeError(f"Import of '{alias.name}' is not permitted in the sandbox.")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in BLOCKED_MODULES:
                raise UnsafeCodeError(f"Import from '{node.module}' is not permitted in the sandbox.")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else None)
            if name in BLOCKED_CALL_NAMES:
                raise UnsafeCodeError(f"Call to '{name}(...)' is not permitted in the sandbox.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in (
                "__init__", "__repr__", "__str__", "__len__"
            ):
                raise UnsafeCodeError(f"Access to dunder attribute '{node.attr}' is not permitted.")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    fn = call.func
                    name = fn.id if isinstance(fn, ast.Name) else None
                    if name == "open":
                        raise UnsafeCodeError("File I/O via open() is not permitted in the sandbox.")


# --------------------------------------------------------------------------- #
# Worker (runs in a fresh subprocess)
# --------------------------------------------------------------------------- #

def _fingerprint(d: Dict[str, Any]) -> str:
    """Content-based fingerprint for a result dict. Deliberately NOT id()-based:
    the sandbox namespace round-trips through cloudpickle between every
    execute_python call (each runs in a fresh subprocess), and unpickling
    always allocates a new object with a new id() -- so an id()-based
    "already recorded" check would forget everything after one step and
    re-capture stale leftover variables as if they were new."""
    import hashlib
    import json
    try:
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        return repr(sorted(d.items(), key=lambda kv: kv[0]))


def _build_fresh_namespace(df, figures_dir: str) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    import scipy
    import scipy.stats as scipy_stats
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pingouin as pg

    from . import stats_toolkit

    results_registry: List[Dict[str, Any]] = []
    recorded_fingerprints: set = set()

    def record_result(name: str, result: Dict[str, Any]) -> None:
        """Attach a result dict to the final report's results tables."""
        entry = {"name": name}
        entry.update(result if isinstance(result, dict) else {"value": result})
        results_registry.append(entry)
        if isinstance(result, dict):
            recorded_fingerprints.add(_fingerprint(result))

    ns: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "pd": pd, "np": np, "scipy": scipy, "stats": scipy_stats,
        "sm": sm, "smf": smf, "plt": plt, "sns": sns, "pg": pg,
        "stats_toolkit": stats_toolkit,
        "df": df,
        "record_result": record_result,
        "_RESULTS": results_registry,
        "_RECORDED_FINGERPRINTS": recorded_fingerprints,
        "FIGURES_DIR": figures_dir,
    }
    return ns


# Namespace keys that are part of the environment, not analysis variables --
# never auto-scanned for "forgotten" results.
_SEED_NAMESPACE_KEYS = {
    "__builtins__", "pd", "np", "scipy", "stats", "sm", "smf", "plt", "sns",
    "pg", "stats_toolkit", "df", "record_result", "_RESULTS",
    "_RECORDED_FINGERPRINTS", "FIGURES_DIR",
}


def _looks_like_toolkit_result(v: Any) -> bool:
    return isinstance(v, dict) and ("test" in v or "effect_size" in v)


def _auto_capture_unrecorded_results(ns: Dict[str, Any], tail_value: Any) -> int:
    """Safety net for a model that computes a stats_toolkit result but
    forgets to call record_result(): scan top-level variables (and the
    last expression's value) for result-shaped dicts not yet recorded
    (by content fingerprint, not id() -- see _fingerprint's docstring for
    why), and record them automatically so the report's Statistical
    Results section isn't empty just because the model skipped a
    bookkeeping call."""
    registry = ns.get("_RESULTS")
    fingerprints = ns.get("_RECORDED_FINGERPRINTS")
    if registry is None or fingerprints is None:
        return 0

    candidates = []
    seen_this_call: set = set()
    for name, val in list(ns.items()):
        if name in _SEED_NAMESPACE_KEYS or name.startswith("_"):
            continue
        if _looks_like_toolkit_result(val):
            fp = _fingerprint(val)
            if fp not in fingerprints and fp not in seen_this_call:
                candidates.append((name, val, fp))
                seen_this_call.add(fp)
    if _looks_like_toolkit_result(tail_value):
        fp = _fingerprint(tail_value)
        if fp not in fingerprints and fp not in seen_this_call:
            candidates.append(("result", tail_value, fp))

    n_added = 0
    for var_name, val, fp in candidates:
        label = val.get("test") or val.get("effect_size") or var_name
        entry = {"name": f"{label} (auto-captured from '{var_name}')"}
        entry.update(val)
        registry.append(entry)
        fingerprints.add(fp)
        n_added += 1
    return n_added


def _save_open_figures(figures_dir: str, fig_counter_start: int) -> List[str]:
    import matplotlib.pyplot as plt
    saved = []
    fignums = plt.get_fignums()
    for i, num in enumerate(fignums):
        fig = plt.figure(num)
        fname = f"fig_{fig_counter_start + i:03d}.png"
        path = os.path.join(figures_dir, fname)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        saved.append(path)
    plt.close("all")
    return saved


def _worker(code: str, state_path: str, df_path: Optional[str], figures_dir: str,
            fig_counter_start: int, queue: "mp.Queue") -> None:
    try:
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                ns = cloudpickle.load(f)
        else:
            with open(df_path, "rb") as f:
                df = cloudpickle.load(f)
            ns = _build_fresh_namespace(df, figures_dir)

        stdout_buf = io.StringIO()
        error_text = None
        last_expr_repr = None
        tail_value = None

        try:
            tree = ast.parse(code, mode="exec")
            body = tree.body
            tail_repr_src = None
            if body and isinstance(body[-1], ast.Expr):
                tail_repr_src = compile(ast.Expression(body[-1].value), "<sandbox_tail>", "eval")
                body = body[:-1]
            main_module = ast.Module(body=body, type_ignores=[])
            compiled = compile(main_module, "<sandbox>", "exec")
            with contextlib.redirect_stdout(stdout_buf):
                exec(compiled, ns)
                if tail_repr_src is not None:
                    tail_value = eval(tail_repr_src, ns)
                    if tail_value is not None:
                        last_expr_repr = repr(tail_value)
        except Exception:
            error_text = traceback.format_exc(limit=6)

        n_auto_captured = 0
        if error_text is None:
            try:
                n_auto_captured = _auto_capture_unrecorded_results(ns, tail_value)
            except Exception:
                pass

        fig_paths = []
        try:
            fig_paths = _save_open_figures(figures_dir, fig_counter_start)
        except Exception:
            pass

        # Persist namespace for the next call (best-effort; drop anything
        # cloudpickle can't serialize rather than failing the whole step).
        try:
            with open(state_path, "wb") as f:
                cloudpickle.dump(ns, f)
        except Exception:
            picklable_ns = {}
            for k, v in ns.items():
                try:
                    cloudpickle.dumps(v)
                    picklable_ns[k] = v
                except Exception:
                    continue
            with open(state_path, "wb") as f:
                cloudpickle.dump(picklable_ns, f)

        n_results = len(ns.get("_RESULTS", []))
        queue.put({
            "success": error_text is None,
            "stdout": stdout_buf.getvalue(),
            "error": error_text,
            "last_expr_repr": last_expr_repr,
            "figures": fig_paths,
            "n_recorded_results": n_results,
            "n_auto_captured": n_auto_captured,
        })
    except Exception:
        queue.put({
            "success": False,
            "stdout": "",
            "error": traceback.format_exc(limit=6),
            "last_expr_repr": None,
            "figures": [],
            "n_recorded_results": 0,
            "n_auto_captured": 0,
        })


# --------------------------------------------------------------------------- #
# Public sandbox interface
# --------------------------------------------------------------------------- #

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    error: Optional[str] = None
    last_expr_repr: Optional[str] = None
    figures: List[str] = field(default_factory=list)
    n_recorded_results: int = 0
    n_auto_captured: int = 0
    rejected_unsafe: Optional[str] = None
    timed_out: bool = False


class CodeSandbox:
    def __init__(self, state_path: str, figures_dir: str, df_seed_path: str, timeout_sec: int = 60):
        self.state_path = state_path
        self.figures_dir = figures_dir
        self.df_seed_path = df_seed_path
        self.timeout_sec = timeout_sec
        self._fig_counter = 1

    def seed_dataframe(self, df) -> None:
        """Write the original dataframe once; the worker loads it on the
        very first execute_python call (after that, namespace state owns it)."""
        with open(self.df_seed_path, "wb") as f:
            cloudpickle.dump(df, f)

    def reset(self) -> None:
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def get_recorded_results(self) -> List[Dict[str, Any]]:
        """Read back every dict passed to record_result() across the whole
        session, from the persisted namespace. Used after the ReAct loop
        ends to build the report's results tables independent of the LLM's
        prose -- the numbers are exactly what the toolkit computed."""
        if not os.path.exists(self.state_path):
            return []
        try:
            with open(self.state_path, "rb") as f:
                ns = cloudpickle.load(f)
            return list(ns.get("_RESULTS", []))
        except Exception:
            return []

    def run(self, code: str) -> ExecutionResult:
        try:
            check_code_safety(code)
        except UnsafeCodeError as e:
            return ExecutionResult(success=False, rejected_unsafe=str(e),
                                    error=f"Rejected by sandbox safety check: {e}")

        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker,
            args=(code, self.state_path, self.df_seed_path, self.figures_dir,
                  self._fig_counter, queue),
        )
        proc.start()
        proc.join(self.timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            return ExecutionResult(success=False, timed_out=True,
                                    error=f"Execution exceeded the {self.timeout_sec}s sandbox timeout "
                                          f"and was terminated. Simplify the analysis (e.g. operate on a "
                                          f"sample, or split into smaller steps).")

        if queue.empty():
            return ExecutionResult(success=False, error="Sandbox worker process crashed with no output "
                                                          "(likely an out-of-memory error or segfault).")

        out = queue.get()
        result = ExecutionResult(
            success=out["success"], stdout=out["stdout"], error=out["error"],
            last_expr_repr=out["last_expr_repr"], figures=out["figures"],
            n_recorded_results=out["n_recorded_results"],
            n_auto_captured=out.get("n_auto_captured", 0),
        )
        self._fig_counter += len(result.figures)
        return result