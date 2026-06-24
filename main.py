#!/usr/bin/env python3
"""Automatic Statistician v2.0 -- CLI entry point.

Example:
    python main.py --data sales.csv --task "Find what predicts customer churn" \\
        --output report.docx

    # Point at a different local server / model:
    python main.py --data sales.csv --task "..." \\
        --base-url http://localhost:11434/v1 --model qwen3:4b

Run `python setup_check.py` first to confirm your local LLM server is
reachable before running a full analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from autostat import __version__
from autostat.config import LLMConfig, AgentConfig, RunPaths
from autostat.data_profiling import load_data, profile_data, profile_to_prompt_string, save_profile
from autostat.llm_client import LLMClient, LLMConnectionError
from autostat.sandbox import CodeSandbox
from autostat.react_agent import run_agent
from autostat.report_generator import build_report
from autostat import validation


def parse_args():
    p = argparse.ArgumentParser(description="Automatic Statistician v2.0 (LLM-based)")
    p.add_argument("--data", required=True, help="Path to a CSV/Excel/JSON/Parquet dataset")
    p.add_argument("--task", required=True, help="What you want analyzed, in plain English")
    p.add_argument("--output", default="report.docx", help="Output .docx path")
    p.add_argument("--output-dir", default=None,
                   help="Working directory for figures/transcript/profile (default: alongside --output)")
    p.add_argument("--model", default=None, help="Model name as known to your local server (default: qwen3:4b)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: http://localhost:11434/v1)")
    p.add_argument("--max-iterations", type=int, default=None, help="Max ReAct loop iterations (default: 20)")
    p.add_argument("--sandbox-timeout", type=int, default=None, help="Per-step sandbox timeout in seconds (default: 60)")
    p.add_argument("--ground-truth", default=None,
                   help="Optional JSON file mapping recorded result names to expected field values")
    p.add_argument("--verbose", action="store_true", help="Print each ReAct step to stdout as it happens")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Automatic Statistician v{__version__}")

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (os.path.dirname(os.path.abspath(args.output)) or ".")
    paths = RunPaths(output_dir=output_dir)

    llm_cfg = LLMConfig()
    if args.model:
        llm_cfg.model = args.model
    if args.base_url:
        llm_cfg.base_url = args.base_url

    agent_cfg = AgentConfig()
    if args.max_iterations:
        agent_cfg.max_iterations = args.max_iterations
    if args.sandbox_timeout:
        agent_cfg.sandbox_timeout_sec = args.sandbox_timeout

    # Catches a partial/mismatched install early (e.g. a zip re-extracted on top of an
    # old folder that skipped some files) with a clear message, instead of an
    # AttributeError mid-run several minutes into a real analysis.
    expected_attrs = {"LLMConfig": (llm_cfg, ["disable_thinking", "stop"]),
                       "AgentConfig": (agent_cfg, ["max_consecutive_parse_failures", "max_repeated_actions"])}
    missing = [f"{cls}.{attr}" for cls, (obj, attrs) in expected_attrs.items()
               for attr in attrs if not hasattr(obj, attr)]
    if missing:
        print(f"Error: your autostat/ package looks out of date or partially overwritten -- "
              f"missing: {', '.join(missing)}.", file=sys.stderr)
        print("Delete the whole project folder and re-extract the zip fresh (don't extract on "
              "top of the old folder -- on Windows, Explorer can silently skip files on "
              "'file already exists' prompts).", file=sys.stderr)
        sys.exit(1)

    print(f"[1/5] Loading and profiling '{args.data}' ...")
    df = load_data(args.data)
    profile = profile_data(df)
    save_profile(profile, paths.profile_path)
    profile_str = profile_to_prompt_string(profile)
    print(f"      {profile['n_rows']} rows x {profile['n_cols']} columns "
          f"({profile['total_missing_cells']} missing cells)")

    print(f"[2/5] Connecting to local LLM server at {llm_cfg.base_url} (model='{llm_cfg.model}') ...")
    try:
        llm_client = LLMClient(llm_cfg)
        if not llm_client.ping():
            print("      Warning: could not verify the server is reachable; will attempt the run anyway.")
    except LLMConnectionError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run `python setup_check.py` for connection troubleshooting.", file=sys.stderr)
        sys.exit(1)

    sandbox = CodeSandbox(state_path=paths.state_path, figures_dir=paths.figures_dir,
                           df_seed_path=os.path.join(paths.workdir, "df_seed.cpkl"),
                           timeout_sec=agent_cfg.sandbox_timeout_sec)
    sandbox.reset()
    sandbox.seed_dataframe(df)

    print(f"[3/5] Running the ReAct agent (max {agent_cfg.max_iterations} steps) ...")
    transcript_lines = []

    def on_step(i, raw, obs):
        line_hdr = f"\n=== Step {i + 1} ==="
        transcript_lines.append(line_hdr)
        transcript_lines.append(f"LLM:\n{raw}")
        transcript_lines.append(f"Observation:\n{obs}")
        if args.verbose:
            print(line_hdr)
            print(raw.strip())
            print(f"-> {obs[:300]}{'...' if len(obs) > 300 else ''}")

    t0 = time.time()
    agent_result = run_agent(args.task, df, profile_str, llm_client, sandbox, agent_cfg, on_step=on_step)
    elapsed = time.time() - t0
    print(f"      Done in {elapsed:.1f}s -- {agent_result.n_iterations} steps, "
          f"{len(agent_result.report_sections)} report section(s), "
          f"finished_cleanly={agent_result.finished_cleanly}")
    if agent_result.aborted_reason:
        print(f"      ABORTED EARLY: {agent_result.aborted_reason}")

    with open(paths.transcript_path, "w") as f:
        f.write(f"# Transcript -- {args.data}\nTask: {args.task}\n")
        f.write("\n".join(transcript_lines))
        f.write(f"\n\nFinal message: {agent_result.finish_message}\n")

    recorded_results = sandbox.get_recorded_results()

    if not recorded_results or not agent_result.report_sections:
        print()
        if not recorded_results and not agent_result.report_sections:
            print("WARNING: zero statistical results AND zero report sections were produced.")
        elif not recorded_results:
            print("WARNING: zero statistical results were recorded (but some report sections exist).")
        else:
            print("WARNING: zero report sections were written (but some statistical results exist).")
        print(f"Full detail in {paths.transcript_path}. Last few raw model replies:")
        for line in transcript_lines[-12:]:
            print(f"  {line}")
        print("\nCommon causes:")
        print("  - The model emitted <think> reasoning that got cut off before any real answer")
        print("    (look for an unclosed <think> tag above). Try raising AUTOSTAT_LLM_MAX_TOKENS,")
        print("    or run setup_check.py to confirm thinking is actually off for your model/server.")
        print("  - The model ran analyses via execute_python but never called record_result() and")
        print("    its output didn't look like a stats_toolkit result dict, so the auto-capture")
        print("    safety net had nothing to grab either -- check whether it's calling")
        print("    stats_toolkit.* functions at all, vs. writing its own from-scratch code.")
        print("  - The model isn't following the Thought/Action/Action Input format at all --")
        print("    re-run with --verbose to watch it live, or try a different/larger model.")
        print()

    print("[4/5] Validating recorded results ...")
    warnings_list = validation.validate_all(recorded_results)
    for w in warnings_list:
        print(f"      - {w}")

    ground_truth_comparison = None
    if args.ground_truth:
        with open(args.ground_truth) as f:
            gt = json.load(f)
        ground_truth_comparison = validation.compare_to_ground_truth(recorded_results, gt)

    executive_summary = None
    if agent_result.report_sections:
        try:
            sections_text = "\n\n".join(f"{s.title}: {s.body}" for s in agent_result.report_sections)
            summary_messages = [
                {"role": "system", "content": "You write concise executive summaries of statistical "
                                               "analyses. Reply with 3-5 plain-English sentences only, "
                                               "no headers, no bullet points."},
                {"role": "user", "content": f"Summarize these findings for a busy reader:\n\n{sections_text}"},
            ]
            executive_summary = llm_client.chat(summary_messages, stop=[]).strip()
        except Exception:
            executive_summary = None  # fall back to the deterministic summary in build_report

    print(f"[5/5] Building report -> {args.output}")
    build_report(
        output_path=args.output,
        dataset_name=os.path.basename(args.data),
        task=args.task,
        df=df,
        profile=profile,
        report_sections=agent_result.report_sections,
        recorded_results=recorded_results,
        figures_dir=paths.figures_dir,
        model_name=llm_cfg.model,
        validation_warnings=warnings_list,
        ground_truth_comparison=ground_truth_comparison,
        executive_summary=executive_summary,
    )
    print(f"\nDone. Report: {args.output}")
    print(f"Transcript: {paths.transcript_path}")
    print(f"Data profile: {paths.profile_path}")


if __name__ == "__main__":
    main()