# Results 4: localizing the unwritten bytes — one bug found and fixed, a second one found and left open

Executed per `RESTORE_CORRUPTION_PLAN4.md`. Branch `restore-corruption`,
continuing from `RESULTS-restore-corruption-3.md`.

## TL;DR

**One real, verified bug found, precisely localized, and fixed. It is not
sufficient on its own — the repro still corrupts.** A second, separate
unwritten-region bug was found in the process and is the most likely
remaining cause; it is not yet fixed. **The corruption is not closed.**
`test_kv_cache_identity_corruption.py` still fails after this session's
patch.

**Bug 1 (fixed):** the Mamba/GDN conv-state window reserves
`num_speculative_blocks` trailing rows (3, matching this deployment's MTP
draft count) for decode-time speculative verification. Nothing ever
writes fresh data into those rows during plain prefill — `preprocess_
mamba`'s boundary-crossing precopy is a pure copy that faithfully carries
forward whatever the source block already held, all the way back to a
request's first block, whose reserved rows are exactly whatever that
physical GPU block's allocation happened to contain. Zero on a virgin
page (harmless); another conversation's stale conv-window bytes on a
block recycled from the pool (corrupts). Fixed in `preprocess_mamba` by
explicitly zeroing those rows on every boundary-crossing copy — but only
*after* the actual precopy dispatch (a real ordering bug in the first cut
of this fix meant it was silently overwritten; see Phase D).
Byte-scan-verified: the previously-universal, precisely `num_spec`-sized
scribble-survival signature in every single Mamba layer is completely
gone post-fix.

**Bug 2 (found, NOT fixed):** a request's very first offloaded chunk
(chunk 0) is stored **entirely unwritten, across every KV-cache group
including attention** — not just Mamba. Same signature as Bug 1 (harmless
when virgin, e.g. every plain unscribbled test in this whole
investigation; corrupts when the GPU pool has been touched by prior
scribbling), but a different location and, since attention is global
(every later token attends to every earlier position), a much larger
blast radius if consumed. This is now the leading suspect for why the
star-gate test still fails after Bug 1's fix. See Anomalies for the
evidence and the next concrete step.

**Star gate (all groups, scribbled, isolated prime → restore): still
corrupts, byte-identical output, before and after Bug 1's fix.** Gates 2+
were not attempted — there is no point running the full regression suite
against a repro that still fails the star gate.

## Phase A — offline byte-scan

**Tooling:** `scan_kv_chunks.py` (repo root), reads the fs-tier's own
`config.json` for group/layer layout (no hardcoded geometry), scans every
stored `.bin` chunk file for maximal runs (`re`-based, vectorized — a
naive byte-by-byte Python loop was ~40x too slow for ~8GB of chunk data)
of a given byte value, and reports them grouped by (KV-cache group,
layer, byte-offset-within-layer, run-length).

**Setup:** fresh fs tier, `VLLM_DEBUG_SCRIBBLE_KV=big` (byte `0x7b` —
finite, non-NaN; fp16 `0x7b7b` = 61280.0, "loud" without NaN's blanket-
propagation confound from session 3), all groups, isolated `c1` priming
(uncontended — matches plan 2/3's proven-clean recipe when unscribbled).
Live reply correct (`'ready'`), confirming the live path is unaffected as
predicted.

```
venv/bin/python scan_kv_chunks.py /d/nvme_cache/vllm_kv --byte 0x7b --min-run 32
```

**Result — completely uniform across all 48 Mamba layer instances (3
groups × 16 layers):**
```
group 0 (16 real layers, layer_slot_bytes=1703936):
  layer 0 (language_model.model.layers.0.linear_attn): 70/70 files affected
      offset=61440 length=61440  in 69/70 files
      ... (chunk 0 shows the whole-file anomaly instead, see below)
[identical pattern, all 16 layers, all 3 groups]

group 3 (17 real layers, layer_slot_bytes=1703936):
  (no surviving runs in real layer data, except chunk 0 — see below)
```

**Control** (unscribbled isolated priming, same recipe minus the env
var): the *identical* byte range reads as zero:
```
venv/bin/python scan_kv_chunks.py /d/nvme_cache/vllm_kv --byte 0x00 --min-run 32
  offset=61440 length=61440  in 5/5 files (every Mamba layer)
  offset=1695744 length=8192  in 5/5 files (the already-known padding tail)
```
Confirms the range is genuinely never written, not a scribble-methodology
artifact.

**Layout, computed from the actual model config (not guessed):**
```python
MambaStateShapeCalculator.gated_delta_net_state_shape(
    tp_world_size=1, num_k_heads=16, num_v_heads=48,
    head_k_dim=128, head_v_dim=128, conv_kernel_size=4, num_spec=3,
) -> conv_shape=(6, 10240), temporal_shape=(48, 128, 128)
MambaStateDtypeCalculator.gated_delta_net_state_dtype(
    model_dtype='bfloat16', mamba_cache_dtype='auto', mamba_ssm_cache_dtype='float16',
) -> conv_dtype=bfloat16, temporal_dtype=float16
```
`conv_bytes = 6 * 10240 * 2 = 122,880`; `temporal_bytes = 48*128*128*2 =
1,572,864`; `122,880 + 1,572,864 = 1,695,744` — exactly the established
unpadded page size. Conv window row 0-2 (`conv_kernel_size - 1 = 3` rows,
bytes `[0:61440)`) are the confirmed history; rows 3-5 (`num_speculative_
blocks = 3` rows, bytes `[61440:122880)`) are the reserved rows — an
**exact** match to the measured gap, both offset and length.

**Attention (group 3): zero surviving bytes in every real chunk** (17
layers, including `mtp.layers.0.self_attn.attn`, confirmed via
`config.json` to be layer index 16, the last one — matching plan 4's
H-MTP structural prediction, but its own KV turns out to be **fully
written**, refuting H-MTP on the consumption side).

**The chunk-0 anomaly** (found, not predicted by the plan): exactly one
chunk per file set — identified by content hash, confirmed via file
mtime to be the very first chunk ever stored in the session — shows the
scribble pattern surviving across its **entire** length, in **every**
group including attention:
```
layer 0 (language_model.model.layers.0.linear_attn): offset=0 length=1695744 in 1/70 files
group 3, layer 0: offset=0 length=28966912 in 1/70 files
```
Same file hash (`fe0c3e21...`) in every group's directory, earliest mtime
of all 280 files by a full second. Not investigated further in Phase A
(out of scope for "the single unwritten range that explains mamba-only-
scribble corruption") but see Anomalies — this turned out to matter.

## Phase B — field bisection

Superseded by the Phase A scan (which directly localizes, rather than
inferring from corrupted-output-text comparison — plan 3's technique).
Attention shows zero unwritten real-chunk bytes; H-MTP is refuted on
these grounds directly, no further Phase B runs needed.

## Phase C — mechanism narrative

**Who never writes the reserved rows, and why:** `preprocess_mamba`
(`v1/worker/mamba_utils.py`) computes `accept_token_bias =
int(input_batch.num_accepted_tokens_cpu[i]) - 1` at every boundary
crossing, and unconditionally resets `num_accepted_tokens_cpu[i] = 1`
after every crossing. For a request that is purely prefilling (no spec-
decode accept/reject ever runs on it), nothing else changes that value,
so `accept_token_bias == 0` at every single prefill-time crossing. With
`token_bias == 0`, `_copy_mamba_state_block`'s conv branch (`state[dst,
:conv_width - token_bias]`) covers the *entire* window, all 6 rows,
including the 3 reserved ones — this is a real, full-width **copy**, not
a skip. But a copy only ever moves forward whatever the source already
held. Nothing, anywhere in the plain-prefill path, ever *originates*
fresh content for those rows — they are copied block-to-block, chunk to
chunk, all the way back to a request's first block, whose value is simply
whatever that physical page's allocation contained.

**Who consumes it:** the same conv-state tensor is read directly by the
Mamba/GDN forward kernel on every subsequent step that uses that block as
its running state — including, critically, the very first decode step of
a *restored* request, whose `preprocess_mamba` call reads `prev_state_idx
= (num_computed_tokens - 1) // block_size` (the cold-resume fallback) and
treats the restored block's *entire* conv window, reserved rows included,
as valid input.

**Why `new_block_ids_to_zero`-style zero-init doesn't cover this:** this
session did not locate a zero-init path that fires for this allocation
route at all — the reserved rows are architecturally *never* the target
of any zero-fill, by design, because the design assumes they only matter
during active speculation, which correctly writes them fresh each time it
happens. The gap is that "never speculated at this position yet" and
"garbage from another conversation" are indistinguishable without an
explicit zero, and nothing supplied one.

## Phase D — fix (partial)

**Fix:** `patches/kv-offload-restore-mamba-conv-spec-reserve.patch`.
`_zero_conv_speculative_reserve()` in `preprocess_mamba`
(`v1/worker/mamba_utils.py`): on every boundary-crossing copy, zero the
destination block's trailing `num_speculative_blocks` conv-window rows.
No-op when `num_speculative_blocks == 0` (no draft config → no reserved
rows exist).

**A real ordering bug in the first implementation, caught by re-scanning
after the "fix" instead of trusting the code:** the first cut placed the
zero-fill call *inside* the per-request staging loop, in program order
right after the precopy's CPU-side bookkeeping (`fused.src_col.np[i] =
...`). But for the fused (spec-decode + hybrid + align — this
deployment's actual path) precopy, that bookkeeping only *stages* values
into CPU-side numpy arrays; the real GPU kernel launch
(`fused.ctx.run_fused_precopy(...)`) happens once, batched, *after* the
entire per-request loop completes. Since my `.zero_()` call executed
eagerly (enqueuing a GPU kernel immediately), it landed on the compute
stream *before* the precopy kernel — so the precopy's copy (from a source
block whose own reserved rows are equally uninitialized) simply
overwrote the zeroing, every time, silently. Caught only by re-running
`scan_kv_chunks.py` against the "fixed" build's own stored output and
finding the exact same 61,440-byte signature, completely unchanged. Fixed
by collecting `(req_state, curr_state_idx)` pairs during the loop and
applying the zero-fill in a second pass *after* the precopy dispatch
(both the fused kernel launch and the non-fused `do_mamba_copy_block`
call). Re-scanned: the signature is now completely gone, matching the
control's unscribbled zero-baseline exactly (see Phase A gate 1 below).

**Gate 1 (star gate) — scribbled recipe, all groups, isolated prime →
restore:**
```
prime c1: prompt=58672 cached=0 60.7s 'ready'          (live, unaffected, as always)
[restart, unscribbled]
turn1: prompt=58672 cached=0 13.6s '大局消ysysres Lifres lif شورres搜ò松 Lif大局...'   <<< CORRUPTED
turn2: prompt=58696 cached=0  5.5s 'fl fed逢 desf煜_CM贞专门的_predエンenres...'          <<< CORRUPTED
```
Confirmed genuine exposure (`GPU had 0 tok`, `70 blk / 58240 tok`) both
turns. **Byte-for-byte identical corrupted text to the pre-fix run.**
Confirmed via re-scan that the Mamba conv-reserve-rows signature is gone
from the stored chunks used in *this exact run* — the fix demonstrably
changed what got stored, and the restored output is unchanged regardless.
**Gate 1 fails. Bug 1's fix, while real, does not explain the repro.**

Gates 2-7 not attempted — no value in running the full regression suite
against a known-still-failing star gate.

## Anomalies & open questions

**The chunk-0 anomaly (Phase A) is the leading remaining suspect,** and
this session found direct supporting evidence for it mattering:

- **Control test:** a *plain, entirely unscribbled* isolated priming,
  replayed with turn 1 alone (forcing chunk 0 through a genuine offload
  restore — `GPU had 0 tok`, `70 blk / 58240 tok`, confirmed) — clean:
  `'ready'`. Same "harmless when virgin, presumably harmful when
  recycled" signature as Bug 1, in a location Bug 1's fix does not touch
  (chunk 0 has `prev_state_idx == -1`, so `preprocess_mamba`'s
  boundary-crossing branch — where the Bug 1 fix lives — never runs for
  it at all).
- **Timing correlation, worth a follow-up trace, not yet conclusive:**
  grepping the star-gate priming's own log for chunk 0's store job
  (`STORE_JOB_BLOCKS job=0 src_block_ids=[1, 5, 9, 13]`) finds a `REALLOC`
  line at the *identical timestamp*: `REALLOC 14 block(s) being
  reallocated this step with NO pending store job registered ...
  blocks=[1, 2, ..., 14]` — block 1, 5, 9, 13 (chunk 0's own blocks, one
  per KV-cache group) are inside that reallocated set. Whether this is
  the actual mechanism (a genuine race between chunk 0's store-job
  creation and its blocks' reallocation, in the same or an adjacent
  step) or a red herring (this log line's own wording allows "never had
  content" as a benign first-use explanation, and this *is* literally
  the first-ever allocation in a fresh process) was not resolved this
  session — time did not allow chasing it to ground the way Bug 1 was
  grounded. **This is the single concrete next step**: instrument
  chunk 0's store specifically (its job-creation time, its actual
  GPU→CPU copy submission and completion time, and what physical
  block/content the copy source pointer actually reads at each of those
  moments) the same way `scan_kv_chunks.py` + the checksum patch
  grounded Bug 1, rather than inferring from log line adjacency.
- Attention's global consumption model (every later token attends to
  every earlier position) makes chunk 0 corruption a much more direct
  explanation for "garbled from the very first token" than the Mamba
  conv-reserve-rows ever was, given Bug 1's fix — which unambiguously
  works, by the same scan methodology — did not change the outcome.

**Updated elimination table** (additions from this session; earlier
sessions' rows stand):

| candidate cause | status | evidence |
|---|---|---|
| H-MTP (MTP draft attention layer's own KV) | **ELIMINATED** | byte-scan: zero surviving bytes in any attention layer, any real chunk, including the mtp layer specifically |
| Mamba conv-state reserved speculative rows | **CONFIRMED, FIXED** | byte-scan before/after; control; code trace; the fix is real and verified, but insufficient alone (star gate still fails) |
| Chunk-0 (first offloaded chunk), all groups | **CONFIRMED PRESENT, NOT YET SHOWN CAUSAL, NOT FIXED** | byte-scan (entire chunk unwritten, every group); control (harmless when virgin, same pattern as bug 1); timing correlation with a REALLOC event is suggestive but not grounded |

**Housekeeping:** server restored to clean default production config, no
debug env vars. `patches/kv-offload-restore-mamba-conv-spec-reserve.patch`
is a real, verified fix — keep it applied (it is: `verify.sh --no-server`
passes). The two debug patches
(`restore-corruption-checksum-debug.patch`,
`restore-corruption-scribble-debug.patch`) remain applied, off by
default. `scan_kv_chunks.py` is a new, reusable, genuinely fast (~1-2 min
for a full 70-chunk/280-file scan) tool — keep it for the chunk-0
follow-up. `/d/nvme_cache` is at 7% (16G/234G) after this session's own
housekeeping (all superseded `vllm_kv_stashed` backups from sessions 1-3
were deleted mid-session with explicit user approval after a stuck-disk
incident); one stash (`vllm_kv.plan4-scribbled`, ~16G, this session's
Phase A reference data) remains and can be deleted once this doc's
excerpts are trusted. `qwen.log.*` snapshots from this session are on
disk, not committed; load-bearing ones are prefixed `plan4-`.

**Do not mark `docs/kv-offload-cache-identity-corruption.md` FIXED.** The
repro still fails.
