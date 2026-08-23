#!/usr/bin/env python3
"""Needle-recall correctness check for the KV offload/restore path.

Embeds a unique 6-digit code deep inside a large first-turn prompt (same
filler style as local_chat_test.py so prefill size is controllable), runs
turn 1 (forcing the prefix to be computed + offloaded), then asks for the
code back in turn 2 at temperature 0. If turn 2 resumes from restored
(RAM/NVMe or GPU-cached) state that is corrupt, the code comes back wrong
or not at all.

Exit code 0 = code recalled exactly; 1 = recall failed.

Usage:
    venv/bin/python needle_recall_test.py [-p 4000] [--turns 2]

--turns 3 adds an extra intermediate turn so the final ask resumes from a
deeper multi-turn state.

Precedent: this exact style of check validated the mamba-align leak fix
(docs/mamba-align-prefill-leak.md).
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:18020"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TOKEN = "88ff2b4ed74b68c225b49457f501d0e4cebd06f59d581e2e"

LOREM_SENTENCE = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
    "eiusmod tempor incididunt ut labore et dolore magna aliqua."
)


def chat(url: str, model: str, token: str, messages: list[dict]) -> tuple[str, float]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "max_tokens": 512,
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
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read())
    dt = time.monotonic() - t0
    return data["choices"][0]["message"]["content"], dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VLLM_URL", DEFAULT_URL))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--token", default=os.environ.get("VLLM_TOKEN", DEFAULT_TOKEN))
    ap.add_argument("-p", "--prompt-size", type=int, default=4000, metavar="N",
                    help="filler sentences around the needle (default 4000)")
    ap.add_argument("--turns", type=int, default=2,
                    help="total turns; the last one asks for the code (default 2)")
    args = ap.parse_args()

    code = f"{random.randint(0, 999999):06d}"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Needle sits ~2/3 into the filler, far from both ends.
    n_before = (args.prompt_size * 2) // 3
    n_after = args.prompt_size - n_before
    system = (
        f"This is a memory test started {now}. A lot of filler follows; "
        "somewhere inside it there is a line with a secret code.\n\n"
        + " ".join([LOREM_SENTENCE] * n_before)
        + f"\n\nTHE SECRET CODE IS {code}. REMEMBER IT.\n\n"
        + " ".join([LOREM_SENTENCE] * n_after)
    )
    messages = [{"role": "system", "content": system}]

    turn_prompts = ["Summarize the material above in one sentence."] * (args.turns - 1)
    turn_prompts.append(
        "What is the secret code embedded in the filler above? "
        "Reply with only the 6 digits."
    )

    for i, prompt in enumerate(turn_prompts, 1):
        messages.append({"role": "user", "content": prompt})
        reply, dt = chat(args.url, args.model, args.token, messages)
        print(f"[turn {i}/{len(turn_prompts)}] {dt:.1f}s: {reply.strip()[:200]!r}")
        messages.append({"role": "assistant", "content": reply})
        if i < len(turn_prompts):
            # Give the offload pipeline a moment to finish stores.
            time.sleep(3)

    final = messages[-1]["content"]
    if code in final:
        print(f"PASS: code {code} recalled exactly")
        return 0
    print(f"FAIL: code {code} NOT found in final reply")
    return 1


if __name__ == "__main__":
    sys.exit(main())
