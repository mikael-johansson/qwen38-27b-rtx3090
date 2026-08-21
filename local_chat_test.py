#!/usr/bin/env python3
"""Minimal local chat REPL against the vLLM OpenAI-compatible endpoint.

No dependencies beyond the stdlib. Talks to http://127.0.0.1:18020 directly
over loopback, streaming, and prints TTFT / total time after each reply so
you can compare against what the request logs and web UI show for the same
prompt -- isolates whether a delay is server-side or client/network-side.

Usage:
    python3 local_chat_test.py
    python3 local_chat_test.py --url http://127.0.0.1:18020 --model qwen3.8-27b
"""

import argparse
import json
import os
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:18020"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TOKEN = "88ff2b4ed74b68c225b49457f501d0e4cebd06f59d581e2e"


def stream_chat(url: str, model: str, token: str, messages: list[dict]) -> list[dict]:
    payload = json.dumps(
        {"model": model, "messages": messages, "stream": True}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )

    t_sent = time.monotonic()
    t_first_byte = None
    t_first_token = None
    t_last_token = None
    full_text = []

    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            now = time.monotonic()
            if t_first_byte is None:
                t_first_byte = now
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning_content")
            if piece:
                if t_first_token is None:
                    t_first_token = now
                t_last_token = now
                full_text.append(piece)
                print(piece, end="", flush=True)

    print()
    ttfb = (t_first_byte - t_sent) if t_first_byte else None
    ttft = (t_first_token - t_sent) if t_first_token else None
    total = (t_last_token or t_first_byte or now) - t_sent
    print(
        f"\n[timing]  ttfb={ttfb:.3f}s  ttft={ttft:.3f}s  total={total:.3f}s"
        if ttfb is not None and ttft is not None
        else "\n[timing]  (no data received)"
    )
    return [{"role": "assistant", "content": "".join(full_text)}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VLLM_URL", DEFAULT_URL))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--token", default=os.environ.get("VLLM_TOKEN", DEFAULT_TOKEN))
    args = ap.parse_args()

    print(f"Connected to {args.url} (model={args.model}). Ctrl-D to quit.\n")
    messages: list[dict] = []
    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            print()
            break
        if not user_input.strip():
            continue
        messages.append({"role": "user", "content": user_input})
        print("assistant> ", end="", flush=True)
        reply = stream_chat(args.url, args.model, args.token, messages)
        messages.extend(reply)


if __name__ == "__main__":
    main()
