# Plan 4: localize the unwritten-but-consumed bytes, name the mechanism, fix it

> **For the executing agent (Opus/Sonnet):** self-contained follow-up to
> plans 1–3 (plan 1 §4 ground rules apply unchanged) and
> `RESULTS-restore-corruption-3.md`. Write findings to
> `RESULTS-restore-corruption-4.md`. Branch `restore-corruption`.
>
> **New ground rule from last session's disk incident:** delete
> superseded `vllm_kv_stashed` backups at the END of each phase (their
> evidence lives in the archived logs); never let `/d/nvme_cache` pass
> ~80% full. Also note for later: the full-disk stuck-`RETRY` lookup
> behavior found last session is a real robustness bug to file
> separately — do not chase it now.

## 0. State of play

Proven (plan 3): the stored chunks contain byte regions the live compute
path never writes but the restore path consumes; virgin-pool zeros are
benign, recycled-block stale bytes (or loud scribble) corrupt. Live
priming output is always correct — only restores break. Mild scribble
(0x3C ≈ 1.06) is inert; loud (0xFF=NaN, 0x7B≈6e4) corrupts. Scribbling
only-mamba corrupts AND scribbling only-attn corrupts (independently).
Text-level comparison cannot localize further. Conv-`token_bias`
remainder ruled out for prefill crossings.

**Working hypothesis to test first (H-MTP), because it fits every
observation:** the stored **attention-group chunks' MTP draft-layer
region** (`mtp.layers.0.self_attn.attn` is the 17th layer in group 3) is
captured before the drafter has ever written KV for those positions —
the drafter's own prefill happens at first decode, *after* the per-chunk
store jobs fire during target prefill. On restore, the drafter consumes
that garbage. Predictions this makes:
- Byte scan (Phase A) finds the mtp layer's region un-overwritten in
  every stored attention chunk.
- Restoring with spec decode disabled is clean (Phase B).
- Magnitude-dependence: zeros/mild values through the drafter are
  benign; huge/NaN destabilize it. (Also implies a second bug: garbage
  drafts should be *rejected*, not corrupt output — Phase C must explain
  why they aren't.)
It does NOT obviously explain the mamba-only-scribble corruption —
Phase A's scan of the mamba chunk files settles whether mamba has its
own unwritten range (possibly spec-decode state columns) or whether the
mamba-selector result had a different cause. Keep both possibilities
open; the scan is objective either way.

## 1. Phase A — offline byte-scan of the stored chunks (no new server instrumentation)

1. Run one **scribbled isolated priming** exactly as plan 3 Phase B:
   fresh fs dir, `VLLM_DEBUG_SCRIBBLE_KV=big` (0x7B), ALL groups, prime
   `c1` turn 1 only. Confirm live reply `'ready'`. Stop the server. The
   fs tier now contains the poisoned truth, at rest.
2. Write `scan_kv_chunks.py` (repo root, standalone): for every `*.bin`
   under the fs dir, find maximal runs of the scribble byte (threshold
   ≥ 32 consecutive bytes) and report `(file, size, [offset, length]...)`
   plus a per-file summary (% scribble-surviving). Group files by size —
   attention chunks (28,966,912 B = 17 regions × 1,703,936 B/layer) vs
   mamba chunks (~26.1 MB: 16 layers × [~1,572,864 B temporal +
   ~61,440 B conv] — derive exact layout constants from the code:
   `MambaSpec.page_size_bytes` composition / the state metadata used by
   `get_mamba_state_copy_func`, and print the layout the scan assumes).
3. Map every surviving range onto that structure: which layer index,
   which sub-tensor, which offset within it. If ranges recur at a fixed
   per-layer period, say so; if exactly one layer region of attention
   chunks survives (prediction: index 16, the mtp layer), H-MTP's first
   prediction is confirmed structurally.
4. **Control:** run the same scan on an *unscribbled* isolated priming's
   fs dir; the same offsets should read as zeros (or as written data
   elsewhere). A range that is scribble in run 1 and zero in run 2 is a
   confirmed never-written-but-stored range.
5. Deliverable: the complete unwritten-range map for both attention and
   mamba chunk files. This is the single most important output of the
   session — everything downstream keys off it.

## 2. Phase B — who consumes it (two cheap discriminating runs)

1. **Spec-off restore:** prime scribbled (0x7B, ALL groups, isolated),
   restart **without** the speculative config (drop `SPEC`/MTP flags for
   this boot only), replay `c1` with full-restore exposure confirmed.
   - Clean → the consumed garbage is spec-decode-only state (drafter KV
     and/or spec state columns) — H-MTP essentially confirmed on the
     consumption side.
   - Still corrupt → a main-path consumer exists; use Phase A's range
     map to identify it (the range says which tensor; grep its readers).
2. **Selector × spec-off cross:** repeat with `SCRIBBLE_KV_ONLY=mamba`
   and `=attn` (still spec-off restore). This resolves plan 3's
   bisection ambiguity: each selector's corruption either survives
   spec-off (main-path consumer in that group) or vanishes (spec-only
   consumer).
3. Record acceptance-rate metrics in the spec-ON corrupted case (the
   accepted-tokens logs): if H-MTP is right, garbage drafter KV should
   crater acceptance — visible confirmation in existing telemetry.

## 3. Phase C — the mechanism narrative

Using Phase A's ranges + Phase B's consumer verdicts:
1. Show the write-timing story for the guilty range(s): for H-MTP, show
   where the drafter writes its KV (first decode / propose path) vs
   when the per-chunk store jobs fire (during target prefill, per
   `_build_store_jobs`); show that local prefix-cache reuse gets the
   later-written rows while the tier snapshot predates them.
2. **If spec-ON restores corrupt output** (they do), explain the second
   bug: why do garbage drafts survive verification? Read the
   rejection/acceptance path for this fork's `draft_sample_method=
   'probabilistic'` at temperature 0 with pathological drafter state
   (NaN/Inf logits → NaN probabilities → acceptance comparisons?). A
   NaN-poisoned acceptance test silently accepting garbage tokens is
   itself a correctness bug to fix (defense: sanitize/reject on
   non-finite draft probs). Trace it concretely; do not hand-wave.
3. If mamba has its own confirmed range+consumer, same treatment.

## 4. Phase D — fix

Order of preference, per range actually found:
1. **Don't store what was never written**: mask the guilty region out of
   the store capture (zero-fill it in the CPU buffer at store time, or
   skip it) so tier bytes are canonical — zeros there are the *proven*
   benign value (c0-virgin behavior was always clean).
2. **Zero-fill on restore** (equivalent effect, restore-side) if the
   store-side mask is awkward for layout reasons.
3. If Phase C(2) confirmed the acceptance-path unsoundness: fix that
   too, separately (sanitize non-finite draft probabilities → treat as
   reject). Two patches, two headers.
4. If a drafter-KV-freshness design fix is cheap (mark restored
   prefixes as drafter-KV-invalid so the drafter re-prefills its own KV
   the way a local first-run does), note it as the *upstream-correct*
   fix in the patch header, but do not build it if (1)+(3) close the
   corruption — scope control.

**Gates (all):**
1. The scribbled recipe (0x7B, all groups, isolated prime → restore,
   spec ON) comes back clean. Star gate — no virgin-pool luck.
2. `test_kv_cache_identity_corruption.py --convs 3 --rounds 2` and
   `--convs 4 --rounds 3`: 3 consecutive clean, exposure-proof runs.
3. Phase A's scan re-run post-fix: zero scribble-surviving ranges in
   stored chunks (or only ranges proven dead/unconsumed, listed).
4. Plan 1 §9 legacy gates: coherence-checking regression script,
   eviction needle test, 15-min 3-way simulate, `verify.sh` green.
5. Acceptance rate after a restore is sane again (compare accepted-
   tokens telemetry before/after fix on the same recipe).
6. Docs: `docs/kv-offload-cache-identity-corruption.md` → FIXED with
   the full mechanism; the restore-corruption RESULTS chain gets a
   closing addendum; draft a short upstream-issue summary (this is
   almost certainly an upstream vLLM bug in offload-connector ×
   MTP/EAGLE × chunked-prefill store timing — the user can decide
   whether to file it).
7. Debug patches reverted from venv; stash dirs cleaned; scratch logs
   pruned per the housekeeping rule.

## 5. Results file format (`RESULTS-restore-corruption-4.md`)

- **TL;DR**: the unwritten-range map in one sentence each; consumer
  verdicts; mechanism; fix; gate status.
- **Phase A**: the range map table (file class, layer/tensor, offset,
  length, scribble-vs-control), plus the scan script's exact invocation.
- **Phase B**: the 2×3 outcome grid (all/mamba/attn × spec-on/spec-off)
  with exposure proof per cell.
- **Phase C**: the write-timing narrative with symbols/files, and the
  acceptance-path trace if applicable.
- **Phase D**: patches + every gate's result.
- **Anomalies/open questions**: as always — raw evidence over
  interpretation; anything the range map shows that this plan didn't
  predict gets quoted verbatim, not smoothed over.
