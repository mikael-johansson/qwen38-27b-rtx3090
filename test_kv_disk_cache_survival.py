#!/usr/bin/env python3
"""Does the NVMe KV-offload cache actually work, and survive a vLLM restart?

Sends a byte-identical large prompt (fixed RNG seed) and reports how much of it
the server could reuse. The only thing that can move the number is the cache.

    venv/bin/python test_kv_disk_cache_survival.py populate   # before restart
    #   ... restart the server ...
    venv/bin/python test_kv_disk_cache_survival.py survived   # after restart
    venv/bin/python test_kv_disk_cache_survival.py control --new   # never-seen prompt

After a restart the GPU and CPU tiers are empty, so anything reused can only
have come from disk. Read the authoritative reuse figure from the NEW REQUEST
line in qwen.log ("N prompt (X% cached, Y uncached tokens)") -- the OpenAI
usage block does not populate prompt_tokens_details here.

Restart survival additionally requires PYTHONHASHSEED to be fixed (micke-start*
sets 0) so block hashes are reproducible across processes, AND a stable cache
directory hash -- see patches/fs-tier-disk-cap.patch and
docs/kv-offload-cache-identity-corruption.md.

MEASURED 2026-08-26 (137,345-token prompt, CTX=long, VISION=1 prefetch):
    same prompt, fresh process : 100.0% cached,  65 uncached,   16.5 s
    never-seen prompt          :   0.0% cached, 137,258 uncached, 187.2 s
    => 11.3x, and the cache demonstrably survives restarts.
"""
import json, os, random, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = (os.environ.get("VLLM_API_KEY")
       or open(os.path.join(HERE, "api_key.txt")).read().strip())
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")

tag = sys.argv[1] if len(sys.argv) > 1 else "run"
words = int(sys.argv[sys.argv.index("--words") + 1]) if "--words" in sys.argv else 30000
seed = 99999999 if "--new" in sys.argv else 20260826


def build_prompt(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = [f"{w}{i}" for i, w in enumerate(
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango".split() * 60)]
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def ask(prompt: str):
    payload = {"model": os.environ.get("VLLM_MODEL", "qwen3.8-27b"),
               "max_tokens": 8, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False},
               "messages": [{"role": "user",
                             "content": prompt + "\n\nReply with the single word: ok"}]}
    req = urllib.request.Request(
        API + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return time.monotonic() - t0, r["usage"]["prompt_tokens"]


if __name__ == "__main__":
    dt, pt = ask(build_prompt(words, seed))
    print(f"[{tag}] prompt={pt} wall={dt:.2f}s  (seed={seed})")
    print("  reuse: grep -a 'NEW REQUEST' qwen.log | tail -1")
