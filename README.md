# Automatic Statistician v2.1 (Groq-powered)

## Overview

Automatic Statistician is an LLM-powered statistical analysis pipeline
that takes a dataset and a plain-English task, automatically performs
statistical analysis, generates visualizations, validates results, and
produces a professional Microsoft Word report.

Unlike previous versions that relied on local LLMs, this version uses
the **Groq API** for fast, high-quality cloud inference. The
architecture remains modular and consists of five layers:

``` text
LAYER 1  data_profiling.py     Load dataset, infer column types, build JSON profile
LAYER 2  sandbox.py            Stateful sandboxed Python execution
         react_agent.py        ReAct (Thought → Action → Observation) agent
LAYER 3  stats_toolkit.py      Statistical analysis toolkit
LAYER 4  report_generator.py   Professional DOCX report generation
LAYER 5  validation.py         Result validation and optional benchmark comparison

main.py                        CLI entry point
```

## Features

-   Automatic dataset profiling
-   Automatic statistical test selection
-   ReAct-based statistical reasoning
-   Safe sandboxed Python execution
-   Assumption checking
-   Automatic chart generation
-   Executive summary generation
-   Word (.docx) report generation
-   Validation against expected results
-   Supports CSV, Excel, JSON and Parquet
-   Powered by Groq LLMs

------------------------------------------------------------------------

# Installation

``` bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

# Configure Groq

Create a `.env` file.

``` env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

AUTOSTAT_LLM_MODEL=llama-3.3-70b-versatile
AUTOSTAT_LLM_TEMPERATURE=0.2
AUTOSTAT_LLM_MAX_TOKENS=4096
AUTOSTAT_LLM_TIMEOUT=180
```

------------------------------------------------------------------------

# Recommended Models

-   llama-3.3-70b-versatile
-   qwen/qwen3-32b
-   openai/gpt-oss-120b
-   meta-llama/llama-4-maverick-17b-128e-instruct

------------------------------------------------------------------------

# Running the Project

``` bash
python main.py ^
    --data wine.csv ^
    --task "Analyze this dataset and generate a statistical report." ^
    --output report.docx
```

Useful flags:

  Flag                Purpose
  ------------------- ---------------------------
  --model             Groq model name
  --max-iterations    Maximum ReAct iterations
  --sandbox-timeout   Sandbox timeout (seconds)
  --ground-truth      Expected benchmark JSON
  --verbose           Print every ReAct step

Outputs:

-   report.docx
-   transcript.md
-   data_profile.json
-   figures/

------------------------------------------------------------------------

# Pipeline

## Layer 1 -- Data Profiling

-   Load CSV, Excel, JSON and Parquet
-   Detect semantic column types
-   Missing value analysis
-   Descriptive statistics
-   Correlation profiling
-   Target-column detection

## Layer 2 -- ReAct Agent

The LLM reasons step by step using a constrained ReAct loop.

Thought → Action → Observation

Actions include:

-   query_data
-   execute_python
-   visualize
-   report
-   finish

The agent executes Python inside a persistent sandbox.

## Layer 3 -- Statistical Toolkit

Includes wrappers for:

-   Descriptive Statistics
-   Correlation Analysis
-   Normality Tests
-   Independent & Paired t-tests
-   Mann--Whitney U
-   Wilcoxon Signed Rank
-   ANOVA
-   Kruskal--Wallis
-   Chi-square
-   OLS Regression
-   Logistic Regression
-   Multinomial Logistic Regression
-   Classification Models
-   Mixed Effects Models
-   Effect Sizes

## Layer 4 -- Report Generator

Automatically creates:

-   Executive Summary
-   Dataset Overview
-   Statistical Results
-   Figures
-   Interpretation
-   Validation
-   Appendix

Output format:

-   Microsoft Word (.docx)

## Layer 5 -- Validation

Validates:

-   Statistical assumptions
-   Ground truth (optional)
-   Result consistency

------------------------------------------------------------------------

# Supported File Formats

-   CSV
-   Excel (.xlsx, .xls)
-   JSON
-   Parquet

------------------------------------------------------------------------

# Troubleshooting

## 503 Over Capacity

Example:

    Error code: 503
    Model is currently over capacity

Possible solutions:

-   Retry after a few seconds
-   Add exponential retry logic
-   Switch to another Groq model

## Invalid API Key

Verify:

``` env
GROQ_API_KEY=...
```

## Empty Response

Increase:

    AUTOSTAT_LLM_MAX_TOKENS

or use a different Groq model.

------------------------------------------------------------------------

# Extending the Toolkit

1.  Add a wrapper in `stats_toolkit.py`
2.  Return a JSON-serializable dictionary
3.  Add the function to `TOOLKIT_CHEAT_SHEET` in `react_agent.py`

The report generator automatically renders new result dictionaries.

------------------------------------------------------------------------

# Project Structure

``` text
autostat/
│
├── config.py
├── data_profiling.py
├── llm_client.py
├── react_agent.py
├── sandbox.py
├── stats_toolkit.py
├── report_generator.py
├── validation.py
│
main.py
README.md
requirements.txt
```

------------------------------------------------------------------------

# Limitations

-   LLM reasoning quality depends on the selected Groq model.
-   API availability depends on Groq service capacity.
-   Statistical recommendations should be reviewed before making
    critical decisions.
-   Sandbox is intended for trusted analysis workloads and is not a
    hardened security boundary.

------------------------------------------------------------------------

# License

MIT License
