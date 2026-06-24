"""Central configuration for Automatic Statistician.

All defaults are tuned for a 4B-parameter model (e.g. Qwen3-4B) served
locally through Ollama on a 4GB-VRAM GPU. Everything here is overridable
via CLI flags (see main.py) or environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class LLMConfig:
    # Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
    # LM Studio uses http://localhost:1234/v1, llama.cpp's `server` uses
    # http://localhost:8080/v1 -- all work unchanged since we use the
    # standard `openai` client pointed at a custom base_url.
    base_url: str = os.environ.get("AUTOSTAT_LLM_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.environ.get("AUTOSTAT_LLM_API_KEY", "ollama")  # unused by Ollama, required by client
    model: str = os.environ.get("AUTOSTAT_LLM_MODEL", "qwen3:4b")
    temperature: float = _env_float("AUTOSTAT_LLM_TEMPERATURE", 0.2)
    max_tokens: int = _env_int("AUTOSTAT_LLM_MAX_TOKENS", 1536)
    # Keep well under the model's loaded context. A 4GB GPU typically can't
    # hold a huge KV cache alongside a 4B model's weights -- 4096-8192 is a
    # safe range. Set the matching num_ctx in your Ollama Modelfile/run flags.
    context_window: int = _env_int("AUTOSTAT_LLM_CONTEXT_WINDOW", 4096)
    request_timeout: int = _env_int("AUTOSTAT_LLM_TIMEOUT", 180)
    stop: list = field(default_factory=lambda: ["\nObservation:", "Observation:"])
    # Qwen3 (and other recent hybrid-reasoning models) think by default,
    # wrapping replies in <think>...</think> before the real answer. That
    # breaks strict Thought/Action/Action Input parsing and burns through a
    # small context window fast. Off by default for this agent loop, which
    # needs short, strictly-formatted turns. See README troubleshooting.
    disable_thinking: bool = os.environ.get("AUTOSTAT_DISABLE_THINKING", "1") != "0"


@dataclass
class AgentConfig:
    max_iterations: int = _env_int("AUTOSTAT_MAX_ITERATIONS", 20)
    sandbox_timeout_sec: int = _env_int("AUTOSTAT_SANDBOX_TIMEOUT", 60)
    max_repeated_actions: int = 3          # loop-detection threshold
    max_consecutive_parse_failures: int = 5  # abort early instead of burning all iterations silently
    max_observation_chars: int = 1800       # truncate long stdout shown back to the LLM
    figures_dirname: str = "figures"
    workdir_dirname: str = "_workdir"


@dataclass
class RunPaths:
    output_dir: str
    figures_dir: str = field(init=False)
    workdir: str = field(init=False)
    state_path: str = field(init=False)
    transcript_path: str = field(init=False)
    profile_path: str = field(init=False)

    def __post_init__(self):
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.workdir = os.path.join(self.output_dir, "_workdir")
        self.state_path = os.path.join(self.workdir, "session_state.cpkl")
        self.transcript_path = os.path.join(self.output_dir, "transcript.md")
        self.profile_path = os.path.join(self.output_dir, "data_profile.json")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.workdir, exist_ok=True)