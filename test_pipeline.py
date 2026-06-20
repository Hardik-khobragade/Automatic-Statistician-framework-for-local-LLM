#!/usr/bin/env python3
"""End-to-end smoke test using a scripted MockLLMClient -- no GPU, no Ollama,
no internet required. Run this first after installing dependencies to
confirm everything (sandbox, stats toolkit, report generation) works on
your machine, independent of whether your local LLM is set up yet.

    python test_pipeline.py

It generates a synthetic dataset, runs a pre-scripted ReAct trace through
the real sandbox and real stats_toolkit functions, and writes a real
test_report.docx you can open to sanity-check formatting.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

from autostat.config import AgentConfig, RunPaths
from autostat.data_profiling import profile_data, profile_to_prompt_string, save_profile
from autostat.llm_client import MockLLMClient
from autostat.react_agent import run_agent
from autostat.report_generator import build_report
from autostat.sandbox import CodeSandbox
from autostat import validation


def make_synthetic_dataset(path: str, n: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.normal(40, 12, n).round(1),
        "income": rng.lognormal(10, 0.5, n).round(2),
        "group": rng.choice(["A", "B", "C"], n),
        "satisfaction": rng.choice([1, 2, 3, 4, 5], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "passed": rng.choice([True, False], n),
    })
    df.loc[df.sample(10, random_state=seed).index, "age"] = np.nan
    df.to_csv(path, index=False)


SCRIPT = [
    'Thought: First look at the columns available.\nAction: query_data\nAction Input: columns',
    'Thought: Get descriptive statistics for the numeric columns.\nAction: execute_python\n'
    'Action Input: res = stats_toolkit.descriptive_stats(df, ["age", "income"])\n'
    'record_result("Descriptive statistics", res)\nprint(res)',
    'Thought: Summarize the descriptives.\nAction: report\nAction Input: Title: Data Overview\n'
    'Age and income both look reasonably distributed, with income showing the right skew typical of income data.',
    'Thought: Visualize the income distribution.\nAction: visualize\nAction Input: fig, ax = plt.subplots()\n'
    'ax.hist(df["income"].dropna(), bins=20)\nax.set_title("Income distribution")',
    'Thought: Test whether income differs between those who passed and did not.\nAction: execute_python\n'
    'Action Input: res = stats_toolkit.ttest_ind(df, "income", "passed")\n'
    'record_result("Income by pass status", res)\nprint(res)',
    'Thought: Test whether income differs across the three groups.\nAction: execute_python\n'
    'Action Input: res = stats_toolkit.anova_oneway(df, "income", "group")\n'
    'record_result("Income by group", res)\nprint(res)',
    'Thought: Write up the comparison findings.\nAction: report\nAction Input: Title: Income Comparisons\n'
    'Neither pass status nor group membership was significantly associated with income in this sample.',
    'Thought: Check whether group and pass status are related.\nAction: execute_python\n'
    'Action Input: res = stats_toolkit.chi_square_test(df, "group", "passed")\n'
    'record_result("Group vs pass status", res)\nprint(res)',
    'Thought: Wrap up.\nAction: report\nAction Input: Title: Conclusion\n'
    'No strong relationships were detected between income, group, and pass status in this synthetic dataset.',
    'Thought: Analysis is complete.\nAction: finish\nAction Input: Analysis complete.',
]


def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_run")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    data_path = os.path.join(test_dir, "synthetic.csv")
    print("Generating synthetic dataset ...")
    make_synthetic_dataset(data_path)

    print("Profiling data (Layer 1) ...")
    df = pd.read_csv(data_path)
    profile = profile_data(df)
    save_profile(profile, os.path.join(test_dir, "profile.json"))
    profile_str = profile_to_prompt_string(profile)

    paths = RunPaths(output_dir=test_dir)
    sandbox = CodeSandbox(state_path=paths.state_path, figures_dir=paths.figures_dir,
                           df_seed_path=os.path.join(paths.workdir, "df_seed.cpkl"), timeout_sec=30)
    sandbox.seed_dataframe(df)

    print("Running the ReAct agent against the real sandbox + stats_toolkit (Layers 2 & 3) ...")
    mock = MockLLMClient(SCRIPT)
    agent_cfg = AgentConfig(max_iterations=15, sandbox_timeout_sec=30)
    result = run_agent("Explore the dataset and test key relationships", df, profile_str, mock, sandbox, agent_cfg)

    assert result.finished_cleanly, "agent did not finish cleanly"
    assert len(result.report_sections) >= 3, "expected at least 3 report sections"
    recorded = sandbox.get_recorded_results()
    assert len(recorded) >= 3, "expected at least 3 recorded statistical results"
    print(f"  OK: {result.n_iterations} steps, {len(result.report_sections)} sections, "
          f"{len(recorded)} recorded results")

    print("Validating results (Layer 5) ...")
    warnings_list = validation.validate_all(recorded)
    print(f"  {len(warnings_list)} validation warning(s)")

    print("Building the .docx report (Layer 4) ...")
    out_path = os.path.join(test_dir, "test_report.docx")
    build_report(
        output_path=out_path, dataset_name="synthetic.csv",
        task="Explore the dataset and test key relationships",
        df=df, profile=profile, report_sections=result.report_sections,
        recorded_results=recorded, figures_dir=paths.figures_dir,
        model_name="mock (offline test)", validation_warnings=warnings_list,
    )
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    print(f"  OK: {out_path} ({os.path.getsize(out_path)} bytes)")

    print("\nAll checks passed. Open the report to see the formatting:")
    print(f"  {out_path}")
    print("\nNext: run `python setup_check.py` to verify your local LLM server, "
          "then `python main.py --data <yourfile> --task \"...\"` for a real analysis.")


if __name__ == "__main__":
    main()
