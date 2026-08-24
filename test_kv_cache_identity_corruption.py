#!/usr/bin/env python3
"""Regression test for silent KV-offload corruption on restore.

Catches the 2026-08-24 class of bug: a conversation whose prefix is evicted
and then restored from the RAM/NVMe offload tier comes back with corrupted
recurrent (Mamba/GDN) state and decodes token soup -- multilingual garbage or
a run of "!!!!!!" -- from the first token, while the request otherwise looks
perfectly healthy (high cache hit rate, normal timings, no warning).

That one was caused by the offload cache directory hash ignoring
mamba_ssm_cache_dtype / mamba_cache_mode, so blocks written under one SSM
layout were loaded and reinterpreted under another. See
docs/kv-offload-cache-identity-corruption.md. This script is a general guard
for the restore path, not just for that cause.

How it works: two ~88K-token conversations, alternating turns. Only about two
fit in the 6.11 GiB GPU pool, so one stays resident (fast, ~1s/turn) while the
other must be restored from the offload tier (slow, 5-15s/turn). The restored
one is where corruption shows. Two conversations are enough to trigger it
deterministically.

    venv/bin/python test_kv_cache_identity_corruption.py
    venv/bin/python test_kv_cache_identity_corruption.py --rounds 6 --convs 4

Exit code 0 if all responses were coherent, 1 if any looked corrupted.

To see the tier movement behind each turn (patches/kv-transfer-request-logging
.patch), watch the log alongside:

    grep -E "KV (LOAD|STORE)" qwen.log
    grep "GPU had 0 tok" qwen.log     # full restores: where a bad block hurts most
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "18020"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("MODEL_NAME", "qwen3.8-27b")

_key_file = REPO / "api_key.txt"
API_KEY = os.environ.get("VLLM_API_KEY") or (
    _key_file.read_text().strip() if _key_file.exists() else ""
)

# Boring, highly compressible prose. The content does not matter -- only that
# each conversation's prefix is long, stable across turns (so it caches and can
# be looked up) and distinct between conversations (so they evict each other
# instead of sharing blocks).
FILLER = (
    "The maintenance log records routine observations from the facility. "
    "Readings are taken at regular intervals and transcribed without "
    "interpretation. Deviations, when they occur, are noted alongside the "
    "ambient conditions and the identity of the technician on duty. "
)

# CJK / Cyrillic / Thai / Hangul. Corrupted decodes reach for these constantly;
# the legitimate answers here are "ready" and "ok".
_SOUP = re.compile(r"[一-鿿Ѐ-ӿ฀-๿가-힯]")


def build_prompt(conv_id: int, target_tokens: int = 86000) -> str:
    marker = f"\n[conversation {conv_id} reference marker {conv_id * 7919}]\n"
    return marker + (FILLER * (target_tokens * 4 // len(FILLER)))


def looks_corrupted(text: str) -> str:
    """Return a reason string if the text looks like corrupted decoding."""
    head = text[:400]
    if re.search(r"!{6,}", head):
        return "run of '!!!!!!'"
    n = len(_SOUP.findall(head))
    if n > 12:
        return f"{n} CJK/Cyrillic/Thai/Hangul chars in first 400"
    return ""


def chat(messages, max_tokens=120, timeout=900):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    msg = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    # Not always populated by vLLM; the authoritative view is the server log.
    details = usage.get("prompt_tokens_details") or {}
    return msg, usage.get("prompt_tokens", 0), details.get("cached_tokens", 0) or 0, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument(
        "--convs",
        type=int,
        default=2,
        help="2 is enough to force eviction; more increases pressure",
    )
    args = ap.parse_args()

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
    except Exception as e:
        print(f"no server on :{PORT} ({e}) -- start one first", file=sys.stderr)
        return 2

    convs = [
        [
            {"role": "system", "content": build_prompt(i)},
            {"role": "user", "content": "Reply with the single word: ready"},
        ]
        for i in range(args.convs)
    ]

    failures = 0
    for rnd in range(args.rounds):
        for i, msgs in enumerate(convs):
            try:
                out, ptok, cached, dt = chat(msgs)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  r{rnd} c{i}  REQUEST FAILED: {e}", flush=True)
                failures += 1
                continue
            why = looks_corrupted(out)
            print(
                f"  r{rnd} c{i}  prompt={ptok:6d} cached={cached:6d} {dt:6.1f}s  "
                f"{out[:52]!r}{'   <<< CORRUPTED' if why else ''}",
                flush=True,
            )
            if why:
                failures += 1
                print(f"        {why}", flush=True)
                print(f"        {out[:300]!r}", flush=True)
            # Grow the conversation so the next turn resumes from a longer,
            # already-cached prefix -- this is what forces the offload lookup.
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": f"Round {rnd}: reply with 'ok'."})

    total = args.rounds * args.convs
    if failures:
        print(f"\nFAIL: {failures}/{total} responses corrupted", flush=True)
        print(
            "Check that every live cache dir carries mamba_* keys:\n"
            "  cat /d/nvme_cache/vllm_kv/*/config.json | grep mamba",
            flush=True,
        )
        return 1
    print(f"\nPASS: {total}/{total} responses coherent", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
