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

    print("Sending a tiny test prompt (this loads the model into VRAM the first time, "
          "which can take a while on a 4GB GPU)...")
    t0 = time.time()
    try:
        reply = client.chat([
            {"role": "system", "content": "Reply with exactly: Thought: ok\nAction: finish\nAction Input: ok"},
            {"role": "user", "content": "test"},
        ])
    except LLMConnectionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"OK: got a response in {elapsed:.1f}s:")
    print("---")
    print(reply.strip()[:300])
    print("---")
    if elapsed > 30:
        print("\nNote: that took a while. For a 4B model on a 4GB GPU, consider:")
        print("  - using a smaller quantization (q4_K_M or smaller)")
        print("  - lowering AUTOSTAT_LLM_CONTEXT_WINDOW / num_ctx (e.g. 4096)")
        print("  - closing other GPU-using applications")
    print("\nYour local LLM setup looks ready. Try main.py next.")


if __name__ == "__main__":
    main()
