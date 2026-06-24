#!/usr/bin/env python3
"""Quick connectivity check for your local LLM server, before running main.py.

Usage:
    python setup_check.py
    python setup_check.py --base-url http://localhost:11434/v1 --model qwen3:4b
"""
from __future__ import annotations

import argparse
import sys
import time

from autostat.config import LLMConfig
from autostat.llm_client import LLMClient, LLMConnectionError
from autostat.react_agent import strip_thinking, parse_action


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()

    cfg = LLMConfig()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model

    print(f"Checking {cfg.base_url} (model='{cfg.model}') ...")
    try:
        client = LLMClient(cfg)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    if not client.ping():
        print("FAILED: server did not respond to a /models list request.")
        print("- Is Ollama / LM Studio / llama.cpp server actually running?")
        print(f"- Is it listening at {cfg.base_url}? (Ollama default: http://localhost:11434/v1)")
        sys.exit(1)
    print("OK: server is reachable.")

    print("Sending a test prompt in the exact ReAct format main.py uses "
          "(this loads the model into VRAM the first time, which can take a "
          "while on a 4GB GPU)...")
    t0 = time.time()
    try:
        reply = client.chat([
            {"role": "system", "content": "Reply with exactly:\nThought: ok\nAction: finish\nAction Input: ok\n"
                                           "No other text, no <think> block."},
            {"role": "user", "content": "Begin."},
        ])
    except LLMConnectionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"OK: got a response in {elapsed:.1f}s. Raw output:")
    print("---")
    print(reply.strip()[:500])
    print("---")

    cleaned, truncated = strip_thinking(reply)
    had_thinking = cleaned != reply.strip()
    parsed = parse_action(cleaned)

    if had_thinking:
        print("\nNote: this model emitted a <think> reasoning block even though we asked it not to,")
        print("and passed `think: false` to the server. This project strips <think> blocks")
        print("automatically, so it's handled -- but it does mean extra tokens are generated on")
        print("every step, which costs time on a 4GB GPU. If you're on Ollama, make sure it's a")
        print("recent version (the `think` request parameter needs server-side support); otherwise")
        print("this is just normal Qwen3 behavior and the automatic stripping takes care of it.")

    if truncated:
        print("\nWARNING: the reply was cut off mid-<think> with no real answer produced at all.")
        print("This means max_tokens (AUTOSTAT_LLM_MAX_TOKENS) is too low for this model's thinking")
        print("output. Either raise it, or confirm thinking mode is truly off for your server/model.")
    elif parsed is None:
        print("\nWARNING: could not parse a Thought/Action/Action Input block out of this reply at all.")
        print("A real run will likely abort early with zero results. Things to try:")
        print("  - a different/smaller model that follows instructions more literally")
        print("  - lowering temperature further (AUTOSTAT_LLM_TEMPERATURE)")
        print("  - checking that your server isn't silently truncating or mangling the system prompt")
    else:
        print("\nOK: the response parses correctly as a ReAct action.")

    if elapsed > 30:
        print("\nNote: that took a while. For a 4B model on a 4GB GPU, consider:")
        print("  - using a smaller quantization (q4_K_M or smaller)")
        print("  - lowering AUTOSTAT_LLM_CONTEXT_WINDOW / num_ctx (e.g. 4096)")
        print("  - closing other GPU-using applications")

    if not truncated and parsed is not None:
        print("\nYour local LLM setup looks ready. Try main.py next.")
    else:
        print("\nResolve the warning(s) above before running a full analysis with main.py --"
              "otherwise expect an early abort with zero recorded results.")


if __name__ == "__main__":
    main()