# Maximising the KV pool, and sharing the card with whisper.cpp

Measured 2026-08-26 .. 2026-08-29 on the RTX 3090 (23.56 GiB usable), MTP path,
`CTX=long`, `MAX_LEN=140000`, with a whisper.cpp STT server co-resident.

[← back to the main README](../README.md) · [gotchas](gotchas.md)

> **RETRACTED 2026-08-29.** Everything below §1–§2 in the original version of
> this document was measured while the vLLM command line was **silently
> truncated** — a malformed continuation in `start_qwen.sh` dropped every flag
> after `${ASYNC_ARGS}`, so those runs had no speculative decoding, no prefix
> caching, no KV offload and no vision args. The larger pools were mostly those
> features being absent, and no soak ever exercised the MTP path that gotcha #4
> is about. Re-measured with the config intact, both headline claims are false:
>
> | claim | reality with the config intact |
> |---|---|
> | "GPU_UTIL is the lever, KV_MEM only caps" | **backwards.** 0.93 and 0.95 both gave 168,913 tokens — GPU_UTIL changed nothing, KV_MEM was the cap |
> | "0.97 → 188,764 tokens, soak-tested" | 0.97 → 168,913, and it **died** under the worst-case load |
>
> The intact config costs ~1.6 GiB more than the crippled one, which is why none
> of the tuned values survived.

**Result, re-measured and verified: `KV_MEM=5606637568` (5347 MiB) →
146,847 tokens, 1.05x at `MAX_LEN=140000`, with STT running.**

Worst case = 4 concurrent prompts of 48882 / 45705 / 42803 / 39757 tokens each
generating 12288, with whisper answering STT throughout:

| KV pool | at rest | peak | headroom | outcome |
|---|---|---|---|---|
| 166,630 tokens *(KV_MEM unset — the original)* | 23030 | 24074 | 502 | **OOM, engine died, 0/4** |
| **146,847 tokens (`KV_MEM` 5347 MiB)** | **22316** | **23544** | **1032** | **4/4, 0 STT failures, healthy** |

**whisper co-residency costs ~19,800 KV tokens.** That is the real trade, and it
is a reduction from the original, not a gain. There is no free capacity here:
the original 166,630-token setting is not safe once whisper holds ~700 MiB.

## 1. `GPU_UTIL` is what sizes the pool — even when `KV_MEM` is set

`gotchas.md` #18 says to pin the pool in bytes rather than tune utilisation, and
`gpu_worker.py:482` logs

> reserved N GiB … as specified by `kv_cache_memory_bytes` config and **skipped
> memory profiling. This does not respect the `gpu_memory_utilization` config.**

That log line is misleading. vLLM takes **`min(KV_MEM, what fits in
util × total after weights and graphs)`**, so:

- `GPU_UTIL` sets the pool.
- `KV_MEM` only caps it.

Evidence: at `GPU_UTIL=0.93`, raising `KV_MEM` from 5519 → 6154 MiB moved
available KV by 0.02 GiB (4.92 → 4.94). Raising `GPU_UTIL` moved it immediately.

`KV_MEM` is still worth setting — it makes the pool reproducible run to run,
which is #18's actual point — but it is not the capacity lever.

## 2. The ladder — VOID, see correction above; being re-measured

Worst case per step: **4 concurrent prompts of 48882 / 45705 / 42803 / 39757
tokens, each generating 12288**, with whisper answering STT continuously.

| `GPU_UTIL` | KV tokens | conc @140k | runtime peak | headroom | outcome |
|---|---|---|---|---|---|
| 0.93 *(old default)* | 158,089 | 1.13x | — | — | — |
| 0.95 | 173,820 | 1.24x | 22510 | 2066 | 4/4, 0 STT fail |
| 0.96 | 180,898 | 1.29x | 22730 | 1846 | 4/4, 0 STT fail |
| **0.97** ← default | **188,764** | **1.35x** | **23010** | **1566** | **4/4, 0 STT fail** |
| 0.98 | 195,842 | 1.40x | 23190 | 1386 | 4/4, 0 STT fail |

No OOMs, no illegal-memory-access events, no engine deaths at any step.

0.97 over 0.98 for margin only: the largest runtime rise ever observed is
1237 MiB, so 0.97 leaves ~330 MiB beyond it against 0.98's ~150.
`GPU_UTIL=0.98` is one env var away.

**Caveat.** A ~25-minute soak cannot prove long-term stability. #4 warns this
path "survives short benchmarks, which is exactly how it fools you", and the
08-28 illegal-access crash took 38 h to surface. These settings are past the
0.93 that #4 soak-tested. `GPU_UTIL=0.93` reverts.

## 3. The startup peak is a transient MTP drafter allocation

Sampling the GPU at 5-20 Hz through startup:

```
t+47.7s  16423 MiB   target weights resident (15.2 GiB)
t+53.0s  23395 MiB   PEAK
t+54.6s  16109 MiB   released 7206 MiB
t+58.6s  21847 MiB   CUDA graphs captured, KV pool reserved
```

The release coincides exactly with `Detected MTP model. Sharing target model
embedding weights` / `lm_head weights`. **It is not a full duplicate of the
checkpoint** (15.59 × 2 could not fit). Only `embed_tokens` and `lm_head`
briefly exist in both forms:

| | |
|---|---|
| both, packed W4A16 (`I32`, `[248320, 1280]`) | 2.36 GiB |
| both, bf16 unpacked (248320 × 5120 × 2 B × 2) | 4.74 GiB |
| coexisting | **7.10 GiB** vs **7.04 GiB measured** |
| the drafter's own `mtp.*` weights | 0.55 GiB |

`vocab_size=248320`, `hidden_size=5120`, `tie_word_embeddings=false` is what
makes those two matrices so large. Confirmed causally: `SPEC=none` drops the
startup peak to **15711 MiB** and the transient disappears entirely.

Three things this peak is **not** sensitive to — the pool is reserved *after* it:

| change | effect on peak |
|---|---|
| `KV_MEM` 5230 → 4096 MiB | **0 MiB** |
| `CG` 32 → 16 | **0 MiB** |
| `--max-num-batched-tokens` 2048 → 1024 | −20 MiB |

So reducing the transient buys no KV capacity; the runtime ceiling binds first.
Its only cost is that whisper must stand aside for ~3 s during startup.

**It is also not reproducible**: measured 23113–23853 MiB across runs, a ~700 MiB
spread. Do not plan margins against a single reading.

## 4. Co-tenancy with whisper.cpp

whisper's VRAM is **not a leak**. Over 1510 requests it plateaus:

| phase | requests | VRAM |
|---|---|---|
| fresh | 0 | 632 MiB |
| identical 4 s clip | 50 → 300 | 694, flat |
| 14 durations (1–30 s) | 356 → 1196 | **724, flat** |
| identical again | 1260 → 1510 | 724, flat |

Lazy per-length working buffers, saturating once it has seen the range of
durations. No periodic restart needed.

**But stop it before starting vLLM.** The startup peak leaves 730 MiB at best
against whisper's 724 MiB plateau, and the peak varies by ~700 MiB run to run.
Once vLLM has settled there is ample room and whisper can come back up.
`whisper.cpp/hermes/start-stack.sh` sequences this.

## 5. Two new gotchas

**A. A killed vLLM leaves a 20 GiB `/dev/shm` segment behind.** The next start
then dies with `OSError: [Errno 14] Bad address` from
`kv_offload/cpu/shared_offload_region.py:110` — the CPU KV tier cannot map its
region. The error mentions neither shared memory nor capacity. Check and clear:

```bash
ls -la /dev/shm/vllm_offload_*.mmap
fuser /dev/shm/vllm_offload_*.mmap        # confirm no live engine holds it
rm -f /dev/shm/vllm_offload_*.mmap
```

**B. vLLM's idle footprint ratchets, but converges.** PyTorch's caching allocator
does not return freed blocks, so "idle" climbs toward the workload's peak demand
and stays there (measured 21986 → 22510 after one worst-case soak, then flat for
120 s at `Running: 0 reqs`). It does **not** grow without bound. The startup
transient is the exception — that memory *is* returned to the driver.

Consequence: a fresh restart looks far healthier than the same server a day
later. Judge headroom from a soaked server, not a freshly started one.

## 6. What did not work

- **Lowering `GPU_UTIL` to buy startup headroom.** With `KV_MEM` set it only
  moves the `request_memory()` gate (`free >= util × total`), and lowering it
  makes that gate *more* permissive. It cannot prevent an OOM at model load.
- **Shrinking `KV_MEM` to lower the startup peak.** Invariant, see §3.
- **A restart timer for whisper.** It plateaus; a timer treats a non-problem.

---

## 7. Starting it: `start-with-whisper.sh`

```bash
./start-with-whisper.sh                  # vLLM then whisper, in order
GPU_UTIL=0.95 ./start-with-whisper.sh    # lower util
./start-with-whisper.sh --no-whisper     # vLLM alone
```

Six steps, each guarding a failure that cost real time here:

1. **stop whisper** — vLLM's `request_memory()` gate needs `GPU_UTIL × total`
   free, and whisper's ~700 MiB breaks it at 0.97. `micke-start-vision.sh` also
   preflights this and prints what to do rather than failing cryptically.
2. **stop vLLM** (TERM then KILL).
3. **wait for the GPU to drain *and hold steady*** — not merely under a
   threshold. Gotcha #2: profiling a still-draining card silently undersizes the
   pool ~25% with no warning.
4. **clear orphaned `/dev/shm/vllm_offload_*.mmap`** — a killed vLLM leaves a
   20 GiB segment and the next start dies with `OSError: [Errno 14] Bad address`.
5. **start vLLM, then VERIFY THE ARG LIST.** A malformed line continuation inside
   the `exec` drops every later flag; the server still comes up healthy, just
   without speculative decoding, prefix caching and KV offload. `sh -n` does not
   catch it. The script diffs the "non-default args" line against a required list
   and fails loudly. **This check is the most important part of the script** —
   without it, a day of measurements was invalid and nobody could tell.
6. **start whisper**, once vLLM is serving.
