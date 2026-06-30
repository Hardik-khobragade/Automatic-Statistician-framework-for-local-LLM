"""Automatic Statistician v2.0 - LLM-based re-implementation.

A local-first pipeline that profiles a dataset, runs a ReAct agent
(backed by a small local LLM such as Qwen3-4B served via Ollama) inside a
restricted code sandbox to perform statistical analysis, and assembles a
Word (.docx) report with embedded charts and tables.
"""

__version__ = "2.1.0"