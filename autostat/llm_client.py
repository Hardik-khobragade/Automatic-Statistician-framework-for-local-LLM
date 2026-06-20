"""LLM client wrapper.

Talks to any OpenAI-compatible local server (Ollama, LM Studio, llama.cpp
`server`, vLLM, etc.) using the standard `openai` Python client pointed at a
custom base_url. Also provides a MockLLMClient with scripted responses,
used by test_pipeline.py to exercise the full pipeline without a GPU.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

from .config import LLMConfig

logger = logging.getLogger("autostat.llm")


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required (pip install openai) -- it is "
                "used purely as an HTTP client for your local OpenAI-compatible "
                "server, no Anthropic/OpenAI account or internet access needed."
            ) from e
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.request_timeout)

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        """Send a chat completion request and return the assistant text."""
        try:
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                stop=stop if stop is not None else self.cfg.stop,
            )
        except Exception as e:
            raise LLMConnectionError(
                f"Could not reach the local LLM server at {self.cfg.base_url} "
                f"(model='{self.cfg.model}'). Is it running? "
                f"Original error: {e}"
            ) from e
        choice = resp.choices[0]
        text = choice.message.content or ""
        return text

    def ping(self) -> bool:
        """Quick connectivity check used by setup_check.py."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


class LLMConnectionError(RuntimeError):
    pass


class MockLLMClient:
    """Deterministic, scripted client for offline pipeline testing.

    `script` is a list of strings; each call to .chat() pops the next one.
    If the script runs out, it returns a 'finish' action so the loop ends
    gracefully instead of raising.
    """

    def __init__(self, script: List[str]):
        self._script = list(script)
        self._i = 0

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        if self._i < len(self._script):
            out = self._script[self._i]
            self._i += 1
            return out
        return "Thought: Out of script, finishing.\nAction: finish\nAction Input: done"

    def ping(self) -> bool:
        return True
