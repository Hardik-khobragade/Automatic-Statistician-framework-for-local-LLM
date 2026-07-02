"""LLM client wrapper.

Talks to any OpenAI-compatible local server (Ollama, LM Studio, llama.cpp
`server`, vLLM, etc.) using the standard `openai` Python client pointed at a
custom base_url. Also provides a MockLLMClient with scripted responses,
used by test_pipeline.py to exercise the full pipeline without a GPU.
"""
from __future__ import annotations

import logging
import sys
from typing import List, Dict, Optional
from groq import Groq
from .config import LLMConfig

logger = logging.getLogger("autostat.llm")


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint."""

    def __init__(self, cfg):
        self.cfg = cfg          # <-- You are missing this

        self._client = Groq(
            api_key=cfg.api_key,
        )

        self.last_diagnostics = {}
        self.cfg.model
        self.cfg.temperature
        self.cfg.max_tokens
        self._client = Groq(
            api_key=cfg.api_key,
        )

    def ping(self)-> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        """Send a chat completion request to Groq and return the assistant response."""

        kwargs = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        if stop is not None and len(stop) > 0:
            kwargs["stop"] = stop

        try:
            resp = self._client.chat.completions.create(**kwargs)

        except Exception as e:
            raise LLMConnectionError(
                f"Could not connect to the Groq API "
                f"(model='{self.cfg.model}').\n"
                f"Original error: {e}"
            ) from e

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = getattr(resp, "usage", None)

        self.last_diagnostics = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "content_chars": len(text),
            "max_tokens_requested": self.cfg.max_tokens,
        }

        if not text.strip():
            print(
                "[llm_client] Warning: Groq returned an empty response.",
                file=sys.stderr,
            )

        return text

    


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