#!/usr/bin/env python3
"""Replay real captured Hermes Agent conversation threads concurrently
against a running vLLM instance, to reproduce the two-large-concurrent-
conversation contention pattern that exposed the offload-eviction-race
bug (see docs/mamba-align-prefill-leak.md's "Offload eviction race"
section) -- without needing to wait for the user's own live traffic.

Uses REAL message arrays captured in requests/*.log (this repo's
request-logging patch writes the full request body per turn), replayed
in original order per thread. Each thread's own turns are sent
sequentially (matching how a real agent can't formulate turn N+1 until
it's seen turn N's response) -- but multiple threads run concurrently
against each other, which is the actual condition that mattered.

Usage:
    venv/bin/python simulate_hermes_traffic.py [--threads N] [--max-turns N]

Requires the server to already be running (micke-start.sh) and, for this
investigation, patches/offload-verbose-eviction-debug-logging.patch to
already be applied + the server restarted to pick it up.
"""

import argparse
import glob
import json
import os
import threading
import time
import urllib.request
from collections import defaultdict

DEFAULT_URL = "http://127.0.0.1:18020"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TOKEN = "88ff2b4ed74b68c225b49457f501d0e4cebd06f59d581e2e"


def load_threads(min_turns: int = 8) -> dict[str, list[list[dict]]]:
    """Group captured requests/*.log files by conversation thread (first
    user message signature), return {signature: [message_array, ...]} in
    original chronological order, for threads with at least min_turns
    captured turns.
    """
    files = sorted(glob.glob("requests/*You_are_Hermes_Agent*.log"))
    threads: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for f in files:
        with open(f, errors="replace") as fh:
            content = fh.read()
        idx = content.find("=== REQUEST ===")
        end = content.find("=== PROMPT ===")
        if idx < 0 or end < 0:
            continue
        try:
            data = json.loads(content[idx + len("=== REQUEST ===") : end].strip())
        except Exception:
            continue
        msgs = data.get("messages", [])
        first_user = next(
            (m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "",
        )
        sig = first_user[:40]
        threads[sig].append((f, msgs))

    out = {}
    for sig, items in threads.items():
        if len(items) < min_turns:
            continue
        items.sort(key=lambda t: t[0])
        msg_lists = [msgs for _, msgs in items]
        # Some signatures (e.g. a repeated "[CONTEXT COMPACTION...]" opener)
        # are reused across many genuinely-different conversation restarts,
        # not one continuously growing thread -- nmsgs jumps around instead
        # of climbing. Only keep signatures where message count is mostly
        # non-decreasing (a real single growing conversation), so replay
        # sends a coherent, ever-longer prefix each turn like the original
        # traffic did.
        nmsgs = [len(m) for m in msg_lists]
        increasing = sum(1 for a, b in zip(nmsgs, nmsgs[1:]) if b >= a)
        if increasing < 0.8 * (len(nmsgs) - 1):
            continue
        if nmsgs[-1] <= nmsgs[0]:
            # No net growth end-to-end -- almost certainly a reused opener
            # template across many distinct restarted conversations, not
            # one real continuously growing thread.
            continue
        out[sig] = msg_lists
    return out


def replay_thread(name: str, turns: list[list[dict]], max_turns: int, url: str, model: str, token: str, log):
    for i, messages in enumerate(turns[:max_turns]):
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "stream": True,
                "max_tokens": 400,
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
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                first_byte = None
                usage = None
                for raw_line in resp:
                    if first_byte is None:
                        first_byte = time.monotonic()
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    d = line[len("data: "):]
                    if d == "[DONE]":
                        break
                    try:
                        chunk = json.loads(d)
                    except Exception:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
            total = time.monotonic() - t0
            ttfb = (first_byte - t0) if first_byte else None
            pt = usage.get("prompt_tokens") if usage else None
            log(
                f"[{name}] turn {i+1}/{min(max_turns, len(turns))} nmsgs={len(messages)} "
                f"prompt_tokens={pt} ttfb={ttfb:.2f}s total={total:.2f}s"
            )
        except Exception as e:
            log(f"[{name}] turn {i+1} FAILED: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VLLM_URL", DEFAULT_URL))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--token", default=os.environ.get("VLLM_TOKEN", DEFAULT_TOKEN))
    ap.add_argument("--threads", type=int, default=2, help="how many concurrent threads to replay")
    ap.add_argument("--max-turns", type=int, default=30, help="max turns to replay per thread")
    ap.add_argument("--min-captured-turns", type=int, default=15)
    args = ap.parse_args()

    all_threads = load_threads(min_turns=args.min_captured_turns)
    if not all_threads:
        print("No threads with enough captured turns found in requests/.")
        return
    # Pick the threads with the most captured turns (most representative
    # of sustained real growth).
    picked = sorted(all_threads.items(), key=lambda kv: -len(kv[1]))[: args.threads]

    print_lock = threading.Lock()

    def log(msg):
        with print_lock:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    print(f"Replaying {len(picked)} thread(s) concurrently:")
    for sig, turns in picked:
        print(f"  {len(turns)} captured turns  {sig!r}")
    print()

    workers = []
    for sig, turns in picked:
        t = threading.Thread(
            target=replay_thread,
            args=(sig[:20], turns, args.max_turns, args.url, args.model, args.token, log),
            daemon=True,
        )
        workers.append(t)
        t.start()
        time.sleep(2)  # stagger starts slightly, matching real arrival pattern

    for t in workers:
        t.join()

    print("\nDone.")


if __name__ == "__main__":
    main()
