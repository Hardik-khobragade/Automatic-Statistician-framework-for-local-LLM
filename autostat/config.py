"""
Central configuration for Automatic Statistician.

Configuration for Groq API.
All values can be overridden using environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


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


# ---------------------------------------------------------------------
# LLM Configuration (Groq)
# ---------------------------------------------------------------------

@dataclass
class LLMConfig:
    # Read from .env
    api_key: str = os.environ.get("GROQ_API_KEY", "")

    # Recommended model
    model: str = os.environ.get(
        "AUTOSTAT_LLM_MODEL",
        "openai/gpt-oss-120b",
    )

    temperature: float = _env_float(
        "AUTOSTAT_LLM_TEMPERATURE",
        0.2,
    )

    max_tokens: int = _env_int(
        "AUTOSTAT_LLM_MAX_TOKENS",
        4096,
    )

    request_timeout: int = _env_int(
        "AUTOSTAT_LLM_TIMEOUT",
        180,
    )

    stop: list = field(
        default_factory=lambda: [
            "\nObservation:",
            "Observation:",
        ]
    )

    # Kept for compatibility with your existing code
    disable_thinking: bool = True


# ---------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------

@dataclass
class AgentConfig:
    max_iterations: int = _env_int(
        "AUTOSTAT_MAX_ITERATIONS",
        20,
    )

    sandbox_timeout_sec: int = _env_int(
        "AUTOSTAT_SANDBOX_TIMEOUT",
        60,
    )

    max_repeated_actions: int = 3

    max_consecutive_parse_failures: int = 5

    max_observation_chars: int = 1800

    figures_dirname: str = "figures"

    workdir_dirname: str = "_workdir"


# ---------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------

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

        self.state_path = os.path.join(
            self.workdir,
            "session_state.cpkl",
        )

        self.transcript_path = os.path.join(
            self.output_dir,
            "transcript.md",
        )

        self.profile_path = os.path.join(
            self.output_dir,
            "data_profile.json",
        )

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.workdir, exist_ok=True)