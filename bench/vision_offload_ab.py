#!/usr/bin/env python3
"""A/B the cost of keeping the ViT tower's weights out of VRAM.

Measures image-encode latency against a running server. Every image is made
unique (a few random pixels) so neither the multimodal processor cache nor the
prefix cache can serve a repeat -- otherwise the second call of an identical
image returns in ~0.2s and hides the ViT cost entirely.

Run it once per backend and compare the medians. Restart the server between
runs -- the backend is chosen at tower construction:

    VISION=1 VISION_OFFLOAD=off      ...   # weights in VRAM (baseline)
    VISION=1 VISION_OFFLOAD=prefetch ...   # streamed per block into a staging pool
    VISION=1 VISION_OFFLOAD=uva      ...   # zero-copy PCIe reads inside every GEMM

Also worth running under concurrent long-context load, not just against an idle
server: the offload connector moves KV blocks over the same PCIe link, and
contention is the main risk to the prefetch backend's overlap. See
patches/vision-tower-cpu-offload.patch.

Usage: venv/bin/python bench/vision_offload_ab.py <tag> [--trials N]
"""
import base64, io, json, os, random, statistics as st, sys, time, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("VLLM_API_KEY") or open(os.path.join(HERE, "..", "api_key.txt")).read().strip()
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
trials = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 5

SIZES = [224, 448, 896, 1280]


def make_unique_image(size: int, rng: random.Random) -> str:
    img = Image.new("RGB", (size, size), (rng.randrange(60, 200),) * 3)
    # unique noise so the processor/prefix caches always miss
    for _ in range(256):
        img.putpixel(
            (rng.randrange(size), rng.randrange(size)),
            (rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(url: str) -> tuple[float, int]:
    payload = {
        "model": os.environ.get("VLLM_MODEL", "qwen3.8-27b"),
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "Reply with the single word: ok"},
        ]}],
    }
    req = urllib.request.Request(
        API + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.monotonic()
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return time.monotonic() - t0, resp["usage"]["prompt_tokens"]


def main():
    rng = random.Random(1234)
    print(f"=== vision A/B [{tag}], {trials} unique images per size ===")
    print(f"{'size':>6} {'vis_tok':>8} {'median_s':>9} {'min_s':>7} {'max_s':>7}")
    for size in SIZES:
        lats, ptoks = [], 0
        for _ in range(trials):
            dt, pt = ask(make_unique_image(size, rng))
            lats.append(dt); ptoks = pt
        print(f"{size:6d} {ptoks:8d} {st.median(lats):9.3f} {min(lats):7.3f} {max(lats):7.3f}")


if __name__ == "__main__":
    main()
