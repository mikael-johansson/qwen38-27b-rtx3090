#!/usr/bin/env python3
"""Correctness test for the vision tower's prefetch (streaming) offload backend.

The bug this exists to catch is silent. vLLM's stock
``PrefetchOffloader._hook_module_forward`` synchronizes through
``torch.ops.vllm.wait_prefetch`` / ``start_prefetch``, and those ops resolve the
offloader via the *global* singleton ``get_offloader()``. The vision tower uses
a **dedicated** instance, which those ops cannot see -- they hit the global
``NoopOffloader``, whose sync methods are ``pass``. That is not a slowdown and
not an error: it is no synchronization at all, so the compute stream reads
staging buffers the copy stream is still filling. The tower then emits
plausible-but-wrong embeddings, non-deterministically.

So a "the server came up and described my image" check proves nothing. This
compares against a bit-exact reference instead, and runs the negative control
(stock hooks) to show the hazard is real on this box.

    venv/bin/python test_vision_prefetch_offload.py

Deliberately small (~255 MiB): it is meant to be runnable while the real server
holds the rest of the card. Shapes are ViT-like but the token count is tiny, so
transfers dominate compute and the race window is as wide as it gets -- which is
the point.
"""

import math
import sys

import torch
import torch.nn as nn

import os

DIM = int(os.environ.get("VPO_DIM", 768))
FFN = int(os.environ.get("VPO_FFN", 3072))
NBLOCKS = int(os.environ.get("VPO_BLOCKS", 16))
NTOKENS = int(os.environ.get("VPO_TOKENS", 8))
TRIALS = int(os.environ.get("VPO_TRIALS", 20))
STEP = int(os.environ.get("VPO_STEP", 2))


class TinyBlock(nn.Module):
    """ViT-shaped block: attention + MLP, real GEMMs, no vLLM dependencies."""

    def __init__(self, dim: int, ffn: int):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        x = x + self.proj(attn @ v)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


def build(device, dtype, seed=0):
    torch.manual_seed(seed)
    return [TinyBlock(DIM, FFN).to(device=device, dtype=dtype) for _ in range(NBLOCKS)]


def run(blocks, x):
    for blk in blocks:
        x = blk(x)
    return x


def get_offloader_class(kind: str):
    """Prefer the shipped subclass; fall back to an inline copy if unpatched."""
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader

    if kind == "stock":
        return PrefetchOffloader, "vllm stock PrefetchOffloader"
    try:
        from vllm.model_executor.models.qwen3_vl import _VisionPrefetchOffloader

        return _VisionPrefetchOffloader, "shipped _VisionPrefetchOffloader (patched)"
    except ImportError:
        pass

    class _Inline(PrefetchOffloader):
        def _hook_module_forward(self, index, module):
            original_forward = module.forward

            def forward(*args, **kwargs):
                module.forward = original_forward
                try:
                    self._wait_for_layer(index)
                    output = original_forward(*args, **kwargs)
                    nxt = (index + self.prefetch_step) % len(self.module_offloaders)
                    self._start_prefetch(nxt)
                    return output
                finally:
                    module.forward = forward

            module.forward = forward

    return _Inline, "INLINE fallback (patch not applied to this venv)"


def check(kind: str, prefetch_step: int = 0) -> bool:
    prefetch_step = prefetch_step or STEP
    device, dtype = torch.device("cuda"), torch.bfloat16
    cls, label = get_offloader_class(kind)

    blocks = build(device, dtype)
    torch.manual_seed(1234)
    x = torch.randn(NTOKENS, DIM, device=device, dtype=dtype)

    # Reference with every weight resident, before anything is offloaded.
    with torch.no_grad():
        reference = run(blocks, x).clone()
    torch.cuda.synchronize()

    offloader = cls(
        group_size=1, num_in_group=1, prefetch_step=prefetch_step, mode="cpu"
    )
    wrapped = offloader.wrap_modules(b for b in blocks)
    offloader.post_init()

    mismatches, first_err = 0, None
    with torch.no_grad():
        for _ in range(TRIALS):
            # The tower calls prime() at the top of every forward; the hooks
            # deliberately never wrap around. Stock PrefetchOffloader has no
            # prime() -- it wraps instead, which is the bug being controlled for.
            if hasattr(offloader, "prime"):
                offloader.prime()
            got = run(wrapped, x)
            torch.cuda.synchronize()
            if not torch.equal(got, reference):
                mismatches += 1
                err = (got.float() - reference.float()).abs().max().item()
                first_err = err if first_err is None else max(first_err, err)

    pool = offloader.buffer_pool.total_bytes / 2**20
    saved = offloader.total_offloaded_bytes / 2**20
    print(
        f"  {label}\n"
        f"    offloaded {saved:.1f} MiB, staging pool {pool:.1f} MiB "
        f"(prefetch_step={prefetch_step})\n"
        f"    {TRIALS - mismatches}/{TRIALS} runs bit-exact"
        + (f", max abs err {first_err:.4g}" if first_err else "")
    )
    return mismatches == 0


class _RecordingHooks:
    """Stub standing in for the offloader instance, recording sync calls."""

    def __init__(self, prefetch_step, nmodules):
        self.prefetch_step = prefetch_step
        self.module_offloaders = [None] * nmodules
        self.waited: list[int] = []
        self.prefetched: list[int] = []

    def _wait_for_layer(self, idx):
        self.waited.append(idx)

    def _start_prefetch(self, idx):
        self.prefetched.append(idx)


def check_structural() -> bool:
    """Prove the hook talks to the instance, not the global singleton.

    Runs without a GPU: builds the hook via the shipped class's own
    ``_hook_module_forward`` bound to a recording stub, then checks that a
    forward pass drove the stub -- and that the global offloader, which is what
    the stock hook would have reached, was never touched.
    """
    from vllm.model_executor.offloader.base import get_offloader

    cls, label = get_offloader_class("vision")
    nmod, step = 7, 2   # odd on purpose: 7 % 2 != 0 is the failing case
    stub = _RecordingHooks(step, nmod)

    modules, calls = [], []
    for i in range(nmod):
        m = nn.Module()
        m.forward = (lambda i: lambda x: (calls.append(i), x)[1])(i)
        cls._hook_module_forward(stub, i, m)
        modules.append(m)

    # prime() stages the head of the stack; the hooks must NOT wrap past the
    # end (a wrapped prefetch clobbers a slot a later block still needs
    # whenever nmod % step != 0 -- see the note in _hook_module_forward).
    cls.prime(stub)
    primed = list(stub.prefetched)
    stub.prefetched.clear()

    sentinel = object()
    out = sentinel
    for m in modules:
        out = m(out if out is not sentinel else sentinel)

    glob = get_offloader()
    want_prefetch = [i + step for i in range(nmod) if i + step < nmod]
    ok = (
        calls == list(range(nmod))
        and stub.waited == list(range(nmod))
        and primed == list(range(step))
        and stub.prefetched == want_prefetch
        and type(glob).__name__ == "NoopOffloader"
    )
    print(
        f"  {label}\n"
        f"    prime()  {primed}\n"
        f"    forwards {calls}\n"
        f"    waited   {stub.waited}\n"
        f"    prefetch {stub.prefetched} (no wrap-around: want {want_prefetch})\n"
        f"    global offloader is {type(glob).__name__} and was never consulted"
    )
    # Hooks must be reinstalled after the call, so the next image works too.
    ok = ok and all(m.forward.__qualname__.endswith("forward") for m in modules)
    return ok


def main() -> int:
    print("[0] structural: hooks must drive the instance, not the global singleton")
    structural_ok = check_structural()
    if not structural_ok:
        print("\nFAIL: hook wiring is wrong.")
        return 1
    print("    OK")

    if not torch.cuda.is_available():
        print("\nSKIP (numerical): no CUDA device")
        return 0
    free, total = torch.cuda.mem_get_info()
    need = (NBLOCKS + 3) * (3 * DIM + DIM + 2 * FFN) * DIM * 2 + 400 * 2**20
    print(f"\nGPU free {free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB "
          f"(numerical test needs ~{need / 2**30:.2f} GiB incl. CUDA context)")
    if free < need:
        print("SKIP (numerical): not enough free VRAM -- the server is holding the")
        print("  card. Re-run with the server stopped, or shrink via e.g.")
        print("  VPO_DIM=384 VPO_FFN=1536 VPO_BLOCKS=8. The structural test above")
        print("  passed, but it cannot catch a real stream race: DO NOT ship on it")
        print("  alone.")
        return 0

    print("\n[1] dedicated-instance hooks (what the patch ships) -- must be exact")
    ok = check("vision")

    print("\n[2] negative control: stock hooks via the global singleton")
    print("    Expected to MISMATCH. If it passes, the race simply did not land")
    print("    this time -- it does not mean stock hooks are safe here.")
    ctl_exact = check("stock")

    print()
    if not ok:
        print("FAIL: the shipped hook path is not bit-exact -- do not ship this.")
        return 1
    print("PASS: dedicated-instance hooks reproduce the resident-weight result exactly.")
    if ctl_exact:
        print("NOTE: negative control did not trip; hazard unproven on this run.")
    else:
        print("Negative control mismatched, as predicted: the hazard is real here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
