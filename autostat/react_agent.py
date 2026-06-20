"""LAYER 2 (part 2): ReAct Agent.

Thought -> Action -> Observation loop, driven by the local LLM. Kept
deliberately simple and constrained for small (~4B) local models:

  - A fixed, small set of actions with an exact text format to parse.
  - query_data is answered directly from the precomputed profile/dataframe
    (no LLM-authored code needed, removes a whole class of failures).
  - execute_python / visualize both run through the same code sandbox.
  - report appends a narrative section; figures generated since the
    previous report call are auto-attached to it (the agent never has to
    name filenames itself).
  - finish ends the loop.
  - Loop detection nudges the model if it repeats an identical action.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import AgentConfig, LLMConfig
from .sandbox import CodeSandbox
from . import stats_toolkit


TOOLKIT_CHEAT_SHEET = """\
Available stats_toolkit functions (call these from execute_python instead of
writing raw statsmodels/scipy code -- they already include the right
assumption checks and effect sizes):
  stats_toolkit.descriptive_stats(df, columns=None)
  stats_toolkit.correlation_matrix(df, columns=None, method="pearson")
  stats_toolkit.test_normality(series, method="shapiro")
  stats_toolkit.ttest_ind(df, value, group)              # 2 independent groups
  stats_toolkit.ttest_paired(df, col1, col2)              # paired/before-after
  stats_toolkit.mann_whitney(df, value, group)
  stats_toolkit.wilcoxon_signed_rank(df, col1, col2)
  stats_toolkit.anova_oneway(df, value, group)            # >=2 groups, with Tukey posthoc
  stats_toolkit.kruskal_wallis(df, value, group)
  stats_toolkit.chi_square_test(df, col1, col2)           # categorical association
  stats_toolkit.ols_regression(df, formula)                # e.g. "y ~ x1 + C(x2)"
  stats_toolkit.logistic_regression(df, formula)           # binary outcome
  stats_toolkit.mixed_effects(df, formula, groups)
  stats_toolkit.cohens_d(series_a, series_b)
  stats_toolkit.cramers_v_from_table(table)

Every one of these returns a dict. To make a result show up in the final
report's results tables, wrap the call:
    res = stats_toolkit.ttest_ind(df, "income", "group")
    record_result("Income by group", res)
    print(res)
"""

SYSTEM_PROMPT_TEMPLATE = """You are an automatic statistician analyzing a dataset for a user.
Work step by step using this exact loop. Output ONLY one Thought/Action/Action Input block per turn, then stop.

Format (follow exactly):
Thought: <your reasoning, 1-3 sentences>
Action: <one of: query_data, execute_python, visualize, report, finish>
Action Input: <input for the action>

Available actions:
- query_data: input is one of "columns", "dtypes", "head", "describe", "missing", "sample:N", "unique:<column>". Returns info about the data, no code needed.
- execute_python: input is python code. `df` (the dataset) is already loaded. Pandas is `pd`, numpy `np`, matplotlib.pyplot `plt`, seaborn `sns`. Variables persist between calls.
- visualize: input is python matplotlib/seaborn code; figures are saved automatically. Variables persist between calls.
- report: input is a short report section, formatted as:
    Title: <section title>
    <body text in plain English, citing concrete numbers you observed>
  Figures created since your last report call are attached to this section automatically.
- finish: input is a one-line wrap-up message. Ends the analysis.

{toolkit_cheat_sheet}

Dataset profile:
{data_profile}

Task: {task}

Guidelines:
- Check relevant statistical assumptions (normality, equal variance) before choosing a parametric test; the toolkit functions already do this for you and report it in the result dict.
- Prefer the toolkit functions over hand-written statistics code.
- Call `report` at least 2-3 times over the course of the analysis (e.g. after descriptives, after each major test) rather than only once at the very end.
- When you have addressed the task, call finish.
- Do not repeat the exact same action twice in a row.

Begin.
"""

EXAMPLE_TRANSCRIPT = """Example of the expected style (different dataset, for illustration only):

Thought: I should first see basic descriptive statistics for the numeric outcome.
Action: execute_python
Action Input: res = stats_toolkit.descriptive_stats(df, ["score"])
record_result("Score descriptives", res)
print(res)

Observation: {{'test': 'descriptive_stats', 'columns': {{'score': {{'n': 120, 'mean': 71.2, ...}}}}}}

Thought: Now I will compare scores between the two treatment groups with a t-test.
Action: execute_python
Action Input: res = stats_toolkit.ttest_ind(df, "score", "treatment")
record_result("Score by treatment", res)
print(res)

Observation: {{'test': 'ttest_ind', 'p_value': 0.031, 'cohens_d': 0.42, ...}}

Thought: The difference is significant with a small-to-medium effect size. I'll write this up.
Action: report
Action Input: Title: Treatment effect on score
The treatment group scored higher on average than the control group (p=0.031, Cohen's d=0.42), a small-to-medium effect. Assumption checks (Levene's test, p>0.05) supported using the standard t-test.

Observation: Section recorded.
"""


@dataclass
class ReportSection:
    title: str
    body: str
    figures: List[str] = field(default_factory=list)


@dataclass
class AgentRunResult:
    transcript: List[Dict[str, str]] = field(default_factory=list)
    report_sections: List[ReportSection] = field(default_factory=list)
    recorded_results: List[Dict[str, Any]] = field(default_factory=list)
    n_iterations: int = 0
    finished_cleanly: bool = False
    finish_message: str = ""


_ACTION_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*Action:\s*(?P<action>\w+)\s*Action Input:\s*(?P<input>.*)",
    re.DOTALL,
)


def parse_action(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse a Thought/Action/Action Input block. Returns (thought, action, input) or None."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    thought = m.group("thought").strip()
    action = m.group("action").strip().lower()
    action_input = m.group("input").strip()
    # If the model kept going and hallucinated its own "Observation:", cut it off.
    action_input = re.split(r"\n\s*Observation:", action_input)[0].strip()
    # Strip markdown code fences if the model wrapped code in them.
    action_input = re.sub(r"^```(?:python)?\s*\n?", "", action_input)
    action_input = re.sub(r"\n?```$", "", action_input).strip()
    return thought, action, action_input


VALID_ACTIONS = {"query_data", "execute_python", "visualize", "report", "finish"}


def handle_query_data(df: pd.DataFrame, query: str) -> str:
    """Answer a query_data action directly, without LLM-authored code."""
    query = query.strip()
    try:
        if query == "columns":
            return ", ".join(df.columns.tolist())
        if query == "dtypes":
            return str(df.dtypes.to_dict())
        if query == "head":
            return df.head(5).to_string()
        if query == "describe":
            return df.describe(include="all").to_string()
        if query == "missing":
            return df.isna().sum().to_string()
        if query.startswith("sample"):
            n = 5
            if ":" in query:
                try:
                    n = int(query.split(":", 1)[1])
                except ValueError:
                    pass
            return df.sample(min(n, len(df))).to_string()
        if query.startswith("unique:"):
            col = query.split(":", 1)[1].strip()
            if col not in df.columns:
                return f"Error: column '{col}' not found. Available columns: {df.columns.tolist()}"
            vc = df[col].value_counts(dropna=False)
            return vc.to_string()
        return (f"Error: unrecognized query_data input '{query}'. "
                f"Use one of: columns, dtypes, head, describe, missing, sample:N, unique:<column>")
    except Exception as e:
        return f"Error answering query_data: {e}"


def parse_report_input(text: str) -> Tuple[str, str]:
    m = re.match(r"\s*Title:\s*(.*?)\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Findings", text.strip()


def run_agent(task: str, df: pd.DataFrame, data_profile_str: str, llm_client,
              sandbox: CodeSandbox, agent_cfg: AgentConfig,
              on_step=None) -> AgentRunResult:
    """Run the ReAct loop to completion (or max_iterations)."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        toolkit_cheat_sheet=TOOLKIT_CHEAT_SHEET,
        data_profile=data_profile_str,
        task=task,
    ) + "\n" + EXAMPLE_TRANSCRIPT

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "Begin the analysis."}]

    result = AgentRunResult()
    recent_actions: List[Tuple[str, str]] = []
    figures_since_last_report: List[str] = []

    for i in range(agent_cfg.max_iterations):
        result.n_iterations = i + 1
        raw = llm_client.chat(messages)
        parsed = parse_action(raw)

        if parsed is None:
            obs = ("Error: could not parse your response. You must reply with exactly:\n"
                   "Thought: ...\nAction: <action_name>\nAction Input: ...")
            result.transcript.append({"role": "assistant", "content": raw})
            result.transcript.append({"role": "observation", "content": obs})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            if on_step:
                on_step(i, raw, obs)
            continue

        thought, action, action_input = parsed
        result.transcript.append({"role": "assistant",
                                   "content": f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"})
        messages.append({"role": "assistant", "content": raw})

        if action not in VALID_ACTIONS:
            obs = f"Error: unknown action '{action}'. Valid actions: {sorted(VALID_ACTIONS)}"
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            result.transcript.append({"role": "observation", "content": obs})
            if on_step:
                on_step(i, raw, obs)
            continue

        # --- loop detection -----------------------------------------------
        sig = (action, action_input[:200])
        recent_actions.append(sig)
        if recent_actions.count(sig) >= agent_cfg.max_repeated_actions:
            obs = ("Error: you have repeated this exact action several times with no new "
                   "result. Try a different analysis, or call finish if you are done.")
            messages.append({"role": "user", "content": f"Observation: {obs}"})
            result.transcript.append({"role": "observation", "content": obs})
            if on_step:
                on_step(i, raw, obs)
            continue

        # --- dispatch -------------------------------------------------------
        if action == "finish":
            result.finished_cleanly = True
            result.finish_message = action_input
            if on_step:
                on_step(i, raw, "(analysis finished)")
            break

        elif action == "query_data":
            obs = handle_query_data(df, action_input)

        elif action in ("execute_python", "visualize"):
            exec_res = sandbox.run(action_input)
            if exec_res.success:
                pieces = []
                if exec_res.stdout.strip():
                    pieces.append(exec_res.stdout.strip())
                if exec_res.last_expr_repr:
                    pieces.append(f"Out: {exec_res.last_expr_repr}")
                if exec_res.figures:
                    pieces.append(f"[saved {len(exec_res.figures)} figure(s)]")
                    figures_since_last_report.extend(exec_res.figures)
                obs = "\n".join(pieces) if pieces else "(executed with no output -- consider adding print() statements)"
            else:
                obs = f"Error: {exec_res.error}"
            if len(obs) > agent_cfg.max_observation_chars:
                obs = obs[:agent_cfg.max_observation_chars] + "\n... (truncated)"

        elif action == "report":
            title, body = parse_report_input(action_input)
            section = ReportSection(title=title, body=body, figures=list(figures_since_last_report))
            figures_since_last_report = []
            result.report_sections.append(section)
            obs = "Section recorded."

        messages.append({"role": "user", "content": f"Observation: {obs}"})
        result.transcript.append({"role": "observation", "content": obs})
        if on_step:
            on_step(i, raw, obs)

    if not result.finished_cleanly:
        result.finish_message = "(reached max iterations without an explicit finish action)"

    return result
