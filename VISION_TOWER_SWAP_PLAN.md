# Swapping the vision tower in and out of VRAM

Branch `vision2`, worktree `/home/mikaeljo/git/vllm/qwen38-27b-vision-swap`
(off `main`, so the running server and the `restore-corruption` work are untouched).

## Verdict

Feasible, and not convoluted — but **not in the form the idea was posed in**, and
the reason matters.

The literal request ("swap it out as soon as it's no longer used, so the KV cache
gets the memory") cannot work, because **vLLM's KV pool is sized once at startup
and never resized**. Freeing VRAM at runtime does not give the KV cache anything.

The good news: once you drop the "swap out on idle" half, what remains is easier
*and* better, and **vLLM 0.27.1 already ships the machinery** — `PrefetchOffloader`
(`vllm/model_executor/offloader/prefetch.py`). The work is ~40 lines in the
existing vision patch, not a new subsystem.

Estimated recovery **~0.71 GiB (~+20,000 tokens of context)** for an image-encode
cost in the low single-digit percent — versus the existing UVA patch's 0.78 GiB
for **+0.3 s to +3.7 s per image**.

---

## 0. Results (2026-08-25) — implemented and measured

Built, measured, and one real bug found and fixed along the way. Numbers below
are measured on this box unless marked otherwise; the predictions further down
are left intact as a record of the reasoning, and where they were wrong it is
called out.

**KV pool** (CTX=long, fp8 KV, `--max-num-seqs 8`, images capped 1024x1024):

| backend | pool | tokens | @140k |
|---|---|---|---|
| off (tower resident) | 5.18 GiB | — | **will not start** |
| off, at MAX_LEN=130000 | 5.18 GiB | 132,680 | 1.02x |
| **prefetch** | **5.90 GiB** | **153,592** | **1.10x** |
| uva | 5.96 GiB | 154,951 | 1.11x |

The headline is stronger than predicted: with the tower resident this
deployment **does not boot at all** at MAX_LEN=140000 — vLLM refuses, needing
5.34 GiB against 5.18 available, and caps the model at 136,000. So streaming is
not an optimization here, it is what makes vision usable at the configured
context length.

**Image latency** (median of 5 unique images, end-to-end incl. 8 output tokens):

| size | vis_tok | off | prefetch | uva |
|---|---|---|---|---|
| 224px | 85 | 0.185s | 0.232s (+0.047) | 0.527s (2.8x) |
| 448px | 217 | 0.286s | 0.330s (+0.044) | 1.380s (4.8x) |
| 896px | 805 | 0.831s | 0.835s (+0.004) | 2.553s (3.1x) |
| 1024px | 1045 | 1.075s | 1.068s (-0.007) | 3.824s (3.6x) |

prefetch costs a fixed ~45 ms on small images — the ~41 ms of pulling 0.7664
GiB across PCIe once — and **nothing measurable at 896px and above**, where each
block's ~30 ms of compute dwarfs its 1.5 ms of transfer. For the full-page OCR
case it is free. The §2 prediction of "1-3% at every size" was right for large
images and wrong for small ones: it used the old patch's encode figures as if
they were pure ViT time, when they were whole-request latencies.

**A third hazard, found only at production scale.** The 16-block test passed
20/20 and the real 27-block tower still failed 0/10. Stock
`_hook_module_forward` wraps its prefetch — `(index + step) % len` — which
aliases a staging slot a later block of the *same* pass still needs whenever
`len % step != 0`. At 27 blocks with step 2, block 25 prefetches block 0 into
slot 0, and block 26 then computes on block 0's weights:

| blocks / step | bit-exact |
|---|---|
| 27 / 2 | **0/6** ← the bug |
| 27 / 3 | 6/6 (3 divides 27) |
| 26 / 2 | 6/6 (2 divides 26) |

Fixed by never wrapping: the hook prefetches only while `index + step < len`,
and `prime()` re-stages the head of the stack at the top of each forward —
correct for any block count and any step. Now 10/10 bit-exact at 27 blocks /
step 2 / 4096 tokens.

End to end this bug showed up as **one misread character** in an OCR
transcription (`DELTA 6` where the resident-weight baseline said `DELTA_6`) —
which reads as "the model is a bit weak at OCR", not as a bug. After the fix the
transcription matches the baseline exactly. That is the whole argument for
testing this against a bit-exact reference rather than by looking at output.

---

## 1. Why "swap out when idle" gains nothing

The chain, all verified in `venv/.../vllm/`:

- `gpu_worker.py:460 determine_available_memory()` runs `memory_profiling()`
  around a dummy forward, then subtracts measured non-KV usage from
  `gpu_memory_utilization * total`.
- `gpu_model_runner.py:7633 initialize_kv_cache_tensors()` allocates the pool
  once from that number.
- There is **no resize path**: `grep -rn "def resize_kv|resize_kv_cache|grow_kv"`
  across the whole package returns nothing.

So the KV pool is a fixed allocation. VRAM you free at 14:05 because no image is
in flight is simply *unused* VRAM — the block pool cannot expand into it.

**The only thing that matters is what is resident at the moment the pool is
sized.** That reframes the goal:

> Don't make the tower *swappable*. Make its steady-state footprint permanently
> tiny, while still running the ViT on the GPU at full speed.

This is strictly easier — no idle detection, no eviction policy, no interaction
with the scheduler, no coordination with the `OffloadingConnector`. And it is
exactly what layer-streaming gives you.

(For completeness: the genuinely dynamic version — carve a reclaimable tail
region out of the KV pool, evict its blocks through the offload connector, unmap
it via the cumem allocator, hand the pages to the ViT, then reverse it — is
possible in principle because `--enable-cumem-allocator` is already on. It needs
a block-pool shrink/grow API that does not exist, scheduler awareness of
temporarily-unavailable blocks, and handling of running requests whose blocks sit
in the sacrificial region. That *is* the "extremely convoluted" option, and it
buys ~0.07 GiB over the simple approach. Not worth it.)

## 2. Recommended design: per-block streaming via `PrefetchOffloader`

Instead of UVA (weights stay in host RAM; GPU reads them across PCIe *during
every GEMM*), stream each block's weights into a small GPU staging buffer just
before that block runs, overlapped with the previous block's compute.

`PrefetchOffloader` already implements precisely this:

- `_CpuParamOffloader.__init__` copies each param to **pinned** host memory and
  frees the VRAM immediately, at construction time — before `load_weights`, which
  then writes into the CPU tensors. Same ordering the LM path already relies on.
- `StaticBufferPool` allocates `prefetch_step` slots of each unique
  `(name, shape, stride, dtype)`; block *N* uses slot `N % prefetch_step`.
- A dedicated `torch.cuda.Stream` does the H2D copies, with a per-layer
  `cuda.Event` so block *N*'s compute waits only on block *N*'s copy.
- Each block's forward is hooked to wait for its own prefetch, then kick off the
  prefetch for block `N + prefetch_step`.

### Why this is so much cheaper than UVA

The UVA patch's own analysis is correct and worth restating, because it explains
the whole gap: GEMM kernels **tile**, so a weight tile is re-read once per
row-block of the activation matrix. A 1280px image re-reads the `qkv` weight on
the order of 50× per layer. Across PCIe at ~20 GB/s that dominates — hence the
cost *scaling with image size* (2.6× at 224px, 3.8× at 1280px).

Streaming reads each weight **exactly once per image**, into VRAM, where the
GEMM's re-reads run at ~936 GB/s as normal.

### The numbers

Measured directly from the checkpoint (`model.safetensors.index.json` +
safetensors headers) — these confirm the old patch's figures exactly:

| | tensors | size |
|---|---|---|
| `visual.blocks.*` (27 blocks) | 324 | **0.7664 GiB** (29.07 MiB/block) |
| `visual.merger.*` | 6 | 0.0835 GiB |
| `visual.patch_embed` / `pos_embed` | 3 | 0.0082 GiB |
| **tower total** | **333** | **0.8582 GiB** |

All `BF16` — the ViT is in the quantizer's `ignore` list (206 entries), so it is
unquantized even though the LM is W4A16.

With only the 27 blocks streamed:

| `prefetch_step` | staging pool | net VRAM recovered |
|---|---|---|
| 2 (recommended) | 58.1 MiB | **0.710 GiB** |
| 3 | 87.2 MiB | 0.681 GiB |

Against the measured `OFF` baseline of 5.17 GiB / 145,326 tokens, 0.710 GiB is
roughly **+20,000 tokens of context**.

### What this actually costs you, in your own numbers

Production today runs `--language-model-only` (`single-user/start_qwen.sh:236`,
unconditional), so **the vision tower is not loaded at all right now**. The
question is therefore what enabling it costs, not what offloading saves.

| config | KV pool | tokens | concurrency @140k | cost vs today |
|---|---|---|---|---|
| today — text-only (measured) | 6.11 GiB | 171,956 | 1.23x | — |
| vision on, tower in VRAM (measured) | 5.17 GiB | 145,326 | 1.04x | -26,630 tok |
| vision on, UVA (measured) | 5.95 GiB | 167,391 | 1.20x | -4,565 tok |
| vision on, streaming (derived) | ~5.89 GiB | ~165,800 | ~1.18x | ~-6,200 tok |

Rows 1-3 are measured (`server_stdout.log` 08-24 for row 1; the vision branch's
A/B for rows 2-3). Row 4 is derived by anchoring on the measured UVA figure and
subtracting the staging pool, at a consistent ~28,300 tokens/GiB across all three
measurements. **Verify row 4 before believing it.**

Resident VRAM breakdown with streaming:

| | GiB |
|---|---|
| merger + patch_embed + pos_embed (not streamed) | 0.0917 |
| staging pool (`prefetch_step=2`) | 0.0568 |
| non-weight mm overhead (`6.11 - 5.95 - 0.0917`) | 0.0683 |
| **total cost of enabling vision** | **~0.217** |

That last row is the honest number, and it is worth being explicit that the
0.0683 GiB is **not** weights — it is the dummy-image encode in the profiling
run, the encoder cache and the mm IPC reservation. No offloading strategy of any
kind removes it.

Note also that **on pure memory, UVA beats streaming** by 0.0568 GiB (~1,600
tokens), because it needs no staging buffer. Streaming's entire advantage is
latency. Choose it because 1,600 tokens is nothing against +0.3-3.7 s per image,
not because it saves more.

Practically, 1.23x -> 1.18x barely moves single-request behaviour (at 140k you
were only ever holding ~1.2 max-length requests). The ~14% larger aggregate pool
vs the naive case (165,800 vs 145,326) matters more for how often 8 concurrent
conversations trigger preemption and offload-cascade churn.

Latency, per block: 29.07 MiB at ~20 GB/s effective pinned H2D ≈ **1.5 ms**.
Compute per block, from the old patch's measured GPU timings:

| image | vis_tok | encode (VRAM) | compute/block | transfer/block | hidden? |
|---|---|---|---|---|---|
| 224px | 85 | 0.193 s | 7.1 ms | 1.5 ms | yes, 4.7× headroom |
| 448px | 217 | 0.456 s | 16.9 ms | 1.5 ms | yes |
| 896px | 805 | 0.923 s | 34.2 ms | 1.5 ms | yes |
| 1280px | 1621 | 1.335 s | 49.4 ms | 1.5 ms | yes |

The transfer hides behind compute at **every** size, with the smallest image
still having ~4.7× headroom. `post_init` issues the first `prefetch_step`
prefetches, so there is no pipeline-fill cost at the start of the forward either.

Residual overhead should be stream-sync bookkeeping only (~27 × ~20 µs ≈ 0.5 ms),
i.e. **1–3%**. That is the number to verify, not to trust — see §5 for the one
thing that could spoil it.

## 3. Your CPU idea — measured, and it does not work

I benchmarked ViT-shaped GEMMs on this actual box (4 threads, leaving headroom
for the running server):

```
tokens=  256   211.5 GFLOP/s  -> 27 blocks   0.99 s
tokens= 1024   223.0 GFLOP/s  -> 27 blocks   3.77 s
tokens= 4096   223.3 GFLOP/s  -> 27 blocks  15.08 s
```

The i7-6700K is **AVX2 only** — no AVX-512, no AMX, and critically **no native
bf16**, so every weight upcasts to fp32 (1.5 GiB resident in RAM instead of
0.77 GiB, plus conversion cost).

Extrapolating to real images, and adding the attention term (the ViT runs full
attention over the patch sequence; at 1280px that is ~6,484 tokens, so
~5.2 TFLOP of QK/AV on top of ~5.3 TFLOP of GEMM):

| image | patches | est. CPU encode |
|---|---|---|
| 224px | ~340 | ~1.3 s |
| 1280px | ~6,484 | **~45–60 s** |

And it is worse than those numbers look: `_execute_mm_encoder` is called
**inline inside `execute_model`** (`gpu_model_runner.py:3668, 3772, 4307`). It is
synchronous. A 45-second CPU encode freezes the entire engine — every other
conversation's decode stalls for the duration. Even the 1.3 s thumbnail case
stalls the server for longer than a whole decode's worth of steps.

You are right that the CPU path also needs the resulting embeddings injected back
into VRAM — but that turns out **not** to be what kills it. The tower emits
`out_hidden_size = 5120` per vision token, so the copy back is tiny:

| image | vis_tok | embeddings | H2D @20 GB/s |
|---|---|---|---|
| 224px | 85 | 0.87 MB | 0.04 ms |
| 1280px | 1621 | 16.60 MB | 0.83 ms |

Under a millisecond even for a full-page image — noise next to a 45-second
encode. So the copy-back is a non-issue; the CPU path fails purely on compute
(and on freezing the engine while it runs). Your conclusion is the right one, and
it is what §2 recommends: keep the ViT on the GPU and stream its *weights* in.
Note the direction of travel is the same either way — ~0.77 GiB of weights once
per image, or ~0.017 GB of embeddings once per image, both over the same PCIe
link. The difference is that streaming weights costs ~41 ms of bus time that
hides entirely behind GPU compute, while the CPU path costs ~45 s of wall clock
that hides behind nothing.

For OCR on full-page images, this is not viable. Streaming costs ~1.5 ms/block
and keeps the ViT on the GPU; the CPU path costs ~1,700 ms/block and stops the
world. Recommend dropping it.

## 4. NVMe — also no

0.77 GiB fits in host RAM, and the transfer must complete *inside* the encode.
NVMe would add a disk read to the critical path for zero memory benefit over
pinned RAM (which is what the prefetcher needs anyway for async H2D). The NVMe
tier earns its place for KV blocks because those are cold, huge, and reusable
across restarts; the ViT weights are none of those.

## 5. Two landmines in reusing vLLM's prefetch path

Both are real and would be easy to miss. The first is a **silent-corruption**
bug, not a perf bug.

### 5a. The prefetch hooks route through the *global* offloader singleton

`PrefetchOffloader._hook_module_forward` does not call itself. It calls
`torch.ops.vllm.wait_prefetch` / `start_prefetch`, and those ops resolve the
offloader through the global singleton:

```python
# prefetch_ops.py:33
def _wait_prefetch_impl(input_tensor, layer_idx): get_offloader()._wait_for_layer(layer_idx)
# prefetch_ops.py:61
def _start_prefetch_impl(output_tensor, layer_idx): get_offloader()._start_prefetch(layer_idx)
```

The existing vision patch deliberately uses a **dedicated** offloader instance
(good reasoning — so the ViT budget can't be raided by LM layers). Carried over
naively to `PrefetchOffloader`, that pattern **silently breaks**: the ops would
hit the global `NoopOffloader`, whose `_wait_for_layer` and `_start_prefetch` are
`pass` (`base.py`). Result: **no synchronization at all** — the compute stream
reads staging buffers while the copy stream is still filling them. That is
non-deterministic garbage image embeddings, not a crash.

**Fix:** don't reuse `_hook_module_forward`. Subclass and install a hook that
calls `self._wait_for_layer(i)` / `self._start_prefetch(next)` on the *instance*,
bypassing `torch.ops`. This is safe here precisely because the ViT is neither
compiled nor cudagraphed — `compile_mm_encoder` and `cudagraph_mm_encoder` both
default to `False` (`config/compilation.py:535,543`) and neither is set in
`single-user/start_qwen.sh`. The `torch.ops` indirection exists only to create
data dependencies for `torch.compile`, which we do not need.

*(If those flags are ever enabled, this design must be revisited — not just this
hook, but the whole event/capture dance.)*

### 5b. `post_init()` never runs for a dedicated instance

`post_init()` is what allocates the buffer pool, calls `sync_cpu_storage()` (to
pick up anything `process_weights_after_loading` changed), and repoints params at
GPU buffers. For the LM path it is called at `gpu_model_runner.py:5572`, after
`load_model`. A dedicated instance gets no such call — and without it the params
are still CPU tensors, so the first ViT forward fails on device mismatch.

**Fix:** call it lazily from `Qwen3_VisionTransformer.forward`, guarded by a
flag. This lands *after* weight loading by construction, and conveniently the
first forward is the dummy profile run — so the 58 MiB buffer pool is allocated
*during* profiling and therefore correctly accounted against the KV pool rather
than silently overcommitting it.

### 5c. Minor: host RAM is tight

`free -g` on this box right now: 31 total, ~3 available, with 20 GiB already
committed to the CPU KV tier (`cpu_bytes_to_use: 21474836480`). Pinning 785 MiB
of ViT weights is not free, and this repo already has a documented incident of
vLLM processes being pushed into swap. Recommend dropping `cpu_bytes_to_use` to
~19 GiB when enabling this.

## 5b. Rebased onto `vision2` (2026-08-26)

This work was originally built on the old `vision` branch, which turned out to
be stale and unstable — it predates most of the `restore-corruption` fixes, and
carries its own failed experiments. It was re-based onto **`vision2`** (= the
`restore-corruption` tip, `ed9c666`), porting only what the streaming
implementation actually needs.

**Ported** (all four were absent from `vision2`):

| file | why necessary |
|---|---|
| `patches/vision-tower-cpu-offload.patch` | the implementation |
| `test_vision_prefetch_offload.py` | the only thing that catches the silent hazards (§7) |
| `bench/vision_offload_ab.py` | latency A/B; unique images so no cache can serve a repeat |
| `VISION_TOWER_SWAP_PLAN.md` | this document |

**Edited surgically** rather than copied, because `vision2`'s versions are the
production ones:

- `single-user/start_qwen.sh` — added only the `VISION=1` toggle
  (`--language-model-only` -> `${LM_ONLY_ARG}`), the `VISION_OFFLOAD` selector,
  and the patch-not-applied guard. Nothing else touched.
- `micke-start-vision.sh` — a **new** launcher rather than editing
  `micke-start.sh`, so the text-only production path is untouched. Same config
  plus `VISION=1`, `VISION_OFFLOAD`, and the multimodal args.

**Deliberately NOT ported from `vision`:**

- **All `--mamba-ssm-cache-dtype` changes.** The old branch moved float16 ->
  bfloat16 -> float32 chasing the `!!!!` loop. `vision2` keeps `float16` and
  that is production's call, not this project's. See the caveat below.
- The `!!!!`-loop investigation commits, `simulate_hermes_traffic.py`,
  `sync-venv.sh`, and the assorted vision experiments.
- The old branch's `micke-start.sh` edits.

> **Open caveat, not addressed here.** `vision2` runs
> `--mamba-ssm-cache-dtype float16` (`single-user/start_qwen.sh:294`). The old
> `vision` branch's commit `233e343` diagnoses fp16 SSM state as the cause of
> non-terminating `!!!!` output: fp16 tops out at 65,504, the recurrent state
> accumulates past it, goes inf -> NaN, and every subsequent token becomes id 0.
> That analysis cites an independent upstream report (syv-ai issue #8) with a
> *different* KV dtype, which isolates the fp16 SSM state as the shared factor —
> and it explicitly notes **images make it more frequent**, because ViT
> embeddings have different magnitudes than text. So this interacts with vision
> specifically. It is left alone because changing it is a production decision
> and is not required for streaming to work. The tell, if it happens: output is
> all `!` from the *first* token, and spec-decode acceptance sits at exactly
> 1.00 / 0.0%. (A *semantic* loop with healthy acceptance is a different bug.)

### Re-measured on `vision2` (2026-08-26)

All figures in §0 were taken on the old `vision` branch. Re-measured on
`vision2`, the KV pool is **better**, because `vision2` keeps the fp16 SSM state
(half the recurrent cache of the old branch's float32), so the same GiB holds
more tokens — 28,065 tok/GiB vs 26,130:

| | pool | tokens | @140k |
|---|---|---|---|
| text-only production (measured earlier, same commit) | 6.11 GiB | 171,956 | 1.23x |
| **vision2 + prefetch** | **5.91 GiB** | **165,869** | **1.18x** |

**Enabling vision now costs 6,087 tokens — 3.5%.** That is very close to the
~6,200 originally derived in §0, and far better than the ~26,600 the old branch
implied. Latency is unchanged (median s, 5 unique images):

| size | vis_tok | vision2 prefetch | old branch prefetch |
|---|---|---|---|
| 224px | 85 | 0.233 | 0.232 |
| 448px | 217 | 0.330 | 0.330 |
| 896px | 805 | 0.835 | 0.835 |
| 1024px | 1045 | 1.079 | 1.068 |

Correctness re-verified on `vision2`: 10/10 bit-exact at real ViT scale
(27 blocks / dim 1152 / 4096 tokens), negative control 0/10 as expected, and the
OCR transcript matches the resident-weight baseline exactly.

## 6. Status

**Implemented, measured, and validated** (2026-08-25). See §0 for results.

| | status |
|---|---|
| patch applies cleanly to the 0.27.1 venv | verified |
| hooks drive the instance, not the global singleton (§5a) | verified; control fails 0/10 |
| no slot aliasing on wrap-around (§0) | verified 6/6 at 27/2, 27/3, 27/4, 26/2 |
| bit-exact vs resident weights, real ViT scale | **10/10** (27 blocks, dim 1152, 4096 tok) |
| `post_init()` lands inside memory profiling (§5b) | verified from startup log ordering |
| KV pool grows | **+0.72 GiB, 132,680 -> 153,592 tokens** |
| image-encode latency cost | **~0 at >=896px; +45 ms fixed on small images** |
| OCR output identical to resident-weight baseline | verified |
| latency under concurrent long-context load | **NOT MEASURED** |

The venv was left **unpatched** — production runs off `restore-corruption`,
which does not carry this patch, so a patched venv would be undetected drift.
To enable:

```bash
patch -p1 -d "$(venv/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')" \
      < patches/vision-tower-cpu-offload.patch
VISION=1 VISION_OFFLOAD=prefetch sh micke-start.sh
```

## 7. Implementation steps

1. **Branch/worktree** — done: `vision-swap` at
   `/home/mikaeljo/git/vllm/qwen38-27b-vision-swap`.
2. **New env var** `VLLM_VISION_PREFETCH_STEP` (int, default `0` = off) in
   `envs.py`, alongside the existing `VLLM_VISION_CPU_OFFLOAD_GB`. Keep both:
   UVA stays available for anyone who wants the extra 0.07 GiB and can eat the
   latency.
3. **`VisionPrefetchOffloader(PrefetchOffloader)`** in the patch:
   - override `_hook_module_forward` to call the instance directly (§5a)
   - construct with `group_size=1, num_in_group=1` (offload every block),
     `prefetch_step` from the env var, `mode="cpu"`
4. **Wire into `Qwen3_VisionTransformer.__init__`** exactly where the UVA branch
   goes today — `wrap_modules()` over the 27 `_make_block(i)` generators. Blocks
   only; leave `patch_embed`, `pos_embed` and `merger` in VRAM (0.09 GiB, and the
   merger is not a repeating unit the pool can share slots for).
5. **Lazy `post_init()`** in `Qwen3_VisionTransformer.forward` (§5b).
6. **Ship as `patches/vision-tower-prefetch-offload.patch`**, following the
   repo's existing patch conventions, with the measured numbers in the header the
   way `vision-tower-cpu-offload.patch` does.

Note `deepstack_visual_indexes` is `[]` on this checkpoint, so
`deepstack_merger_list` is empty and the deepstack branch in the forward loop is
dead — one less thing to worry about.

## 8. Validation

- **Correctness first, and specifically against §5a.** Encode the same image
  with the feature off and on and compare the embeddings — a missing-sync bug
  produces *plausible but wrong* output, so an eyeball check on the generated
  text is not sufficient. Run it 20× and check for run-to-run variation; a
  correct implementation is bit-identical every time.
- **KV pool delta** from the startup log line (expect ~+0.71 GiB / ~+20k tokens).
- **Latency A/B** with `bench/vision_offload_ab.py`, which already exists on the
  `vision` branch and uses 5 unique images per size so neither the mm-processor
  cache nor the prefix cache can serve a repeat. Extend it to a three-way
  OFF / UVA / PREFETCH comparison.
- **The one risk worth watching:** PCIe contention. The `OffloadingConnector` is
  continuously moving KV blocks between VRAM, the 20 GiB CPU tier and NVMe over
  the *same* x16 link. The 1.5 ms/block headroom calculation assumes a quiet bus.
  Measure image latency under concurrent long-context load, not just on an idle
  server — that is where this design would show its worst case, and `prefetch_step=3`
  (87 MiB, buys 2 blocks of slack instead of 1) is the mitigation if it does.
