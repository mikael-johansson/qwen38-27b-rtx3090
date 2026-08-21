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
import datetime
import json
import os
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:18020"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TOKEN = "88ff2b4ed74b68c225b49457f501d0e4cebd06f59d581e2e"

LOREM_SENTENCE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua."
)


def build_test_system_prompt(n: int) -> str:
    """A system prompt guaranteed to miss the prefix cache every time
    (the timestamp changes every call), followed by N filler sentences so
    the size is controllable for prefill-throughput testing.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preamble = (
        f"This is a prompt processing test started {now}. A lot of filler "
        "will now arrive. You do not need to act on it, it is only there "
        "to test prompt processing."
    )
    filler = " ".join([LOREM_SENTENCE] * n)
    return f"{preamble}\n\n{filler}"


def stream_chat(url: str, model: str, token: str, messages: list[dict]) -> list[dict]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            # Ask vLLM for a final usage-only chunk (empty choices, populated
            # "usage") so we can compute real PP/gen tok/s, not just timing.
            "stream_options": {"include_usage": True},
        }
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
    usage = None

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
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            piece = delta.get("content") or delta.get("reasoning")
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

    parts = []
    if ttfb is not None and ttft is not None:
        parts.append(f"ttfb={ttfb:.3f}s")
        parts.append(f"ttft={ttft:.3f}s")
        parts.append(f"total={total:.3f}s")
    else:
        print("\n[timing]  (no data received)")
        return [{"role": "assistant", "content": "".join(full_text)}]

    if usage:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens and ttft > 0:
            parts.append(f"pp={prompt_tokens / ttft:.1f}tok/s")
        if (
            completion_tokens
            and completion_tokens > 1
            and t_first_token is not None
            and t_last_token is not None
            and t_last_token > t_first_token
        ):
            gen_s = t_last_token - t_first_token
            parts.append(f"gen={(completion_tokens - 1) / gen_s:.1f}tok/s")
        parts.append(f"prompt={prompt_tokens} completion={completion_tokens}")

    print("\n[timing]  " + "  ".join(parts))
    return [{"role": "assistant", "content": "".join(full_text)}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VLLM_URL", DEFAULT_URL))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--token", default=os.environ.get("VLLM_TOKEN", DEFAULT_TOKEN))
    ap.add_argument(
        "-p",
        "--prompt-size",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Prefill-throughput test mode: prepend a fresh, always-cold "
            "system prompt (unique timestamp + N filler sentences) to "
            "every turn instead of a growing conversation, so each request "
            "measures raw uncached PP tok/s."
        ),
    )
    args = ap.parse_args()

    print(f"Connected to {args.url} (model={args.model}). Ctrl-D to quit.\n")
    if args.prompt_size:
        print(f"Prompt-processing test mode: {args.prompt_size} filler sentences per turn.\n")
    messages: list[dict] = []
    while True:
        try:
            user_input = input("you> ")
        except EOFError:
            print()
            break
        if not user_input.strip():
            continue
        if args.prompt_size:
            # Standalone each turn: a changing system prompt at the front
            # would poison any accumulated history's cache anyway, so
            # there's no point carrying it forward.
            messages = [
                {"role": "system", "content": build_test_system_prompt(args.prompt_size)},
                {"role": "user", "content": user_input},
            ]
        else:
            messages.append({"role": "user", "content": user_input})
        print("assistant> ", end="", flush=True)
        reply = stream_chat(args.url, args.model, args.token, messages)
        if not args.prompt_size:
            messages.extend(reply)


if __name__ == "__main__":
    main()
