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
        self.last_diagnostics: Dict[str, object] = {}

    def chat(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None) -> str:
        """Send a chat completion request and return the assistant text."""
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stop=stop if stop is not None else self.cfg.stop,
        )
        # Ollama's OpenAI-compatible endpoint accepts a non-standard `think`
        # field on some versions to suppress Qwen3/DeepSeek-R1-style
        # reasoning output. Harmless to send even if the server ignores it;
        # if a stricter server rejects unknown fields outright, retry without it.
        # getattr(..., True) rather than self.cfg.disable_thinking: if someone is running
        # with a slightly stale/partially-overwritten config.py (e.g. a zip re-extracted
        # on top of an old folder didn't replace every file), default to disabling
        # thinking mode rather than crashing -- that's the safer default for this agent.
        think_param_sent = getattr(self.cfg, "disable_thinking", True)
        if think_param_sent:
            kwargs["extra_body"] = {"think": False}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "extra_body" in kwargs:
                kwargs.pop("extra_body")
                think_param_sent = "rejected_by_server"
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as e2:
                    raise LLMConnectionError(
                        f"Could not reach the local LLM server at {self.cfg.base_url} "
                        f"(model='{self.cfg.model}'). Is it running? Original error: {e2}"
                    ) from e2
            else:
                raise LLMConnectionError(
                    f"Could not reach the local LLM server at {self.cfg.base_url} "
                    f"(model='{self.cfg.model}'). Is it running? Original error: {e}"
                ) from e

        choice = resp.choices[0]
        text = choice.message.content or ""

        # Some OpenAI-compatible servers (Ollama, and the DeepSeek-style
        # convention many others copied) return reasoning in a separate,
        # non-standard field instead of (or in addition to) `content` --
        # commonly `reasoning_content` or `reasoning`. The official openai
        # client doesn't define these, but pydantic models built with
        # extra="allow" still expose them. We don't feed this into the
        # parser, but we surface it so an empty/odd `content` is explainable
        # instead of a silent mystery.
        reasoning_text = None
        for attr in ("reasoning_content", "reasoning", "thinking"):
            val = getattr(choice.message, attr, None)
            if val:
                reasoning_text = val
                break

        usage = getattr(resp, "usage", None)
        self.last_diagnostics = {
            "think_param_sent": think_param_sent,
            "finish_reason": getattr(choice, "finish_reason", None),
            "content_chars": len(text),
            "reasoning_field_present": reasoning_text is not None,
            "reasoning_chars": len(reasoning_text) if reasoning_text else 0,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "max_tokens_requested": self.cfg.max_tokens,
        }

        if not text.strip():
            d = self.last_diagnostics
            hit_token_limit = d["finish_reason"] in ("length", "max_tokens") or (
                d["completion_tokens"] is not None and d["completion_tokens"] >= self.cfg.max_tokens
            )
            print(
                f"[llm_client] empty content returned -- finish_reason={d['finish_reason']}, "
                f"completion_tokens={d['completion_tokens']}/{d['max_tokens_requested']}, "
                f"think_param_sent={d['think_param_sent']}, "
                f"separate_reasoning_field_present={d['reasoning_field_present']} "
                f"({d['reasoning_chars']} chars)."
                + (" The model used its ENTIRE token budget and produced zero real-answer tokens -- "
                   "almost certainly still thinking internally despite think:false. Raise "
                   "AUTOSTAT_LLM_MAX_TOKENS substantially, and/or upgrade Ollama (run `ollama --version` -- "
                   "the `think` parameter needs a fairly recent release), and/or try a model without "
                   "forced reasoning (e.g. qwen2.5:3b-instruct, llama3.2:3b) instead of qwen3."
                   if hit_token_limit else ""),
                file=sys.stderr,
            )
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