# Plan 2: restore corruption — split store-side vs restore-side, then drill

> **For the executing agent (Opus/Sonnet):** self-contained follow-up to
> `RESTORE_CORRUPTION_PLAN.md` (whose §4 ground rules apply unchanged —
> reread them) and `RESULTS-restore-corruption.md`. Write findings to
> `RESULTS-restore-corruption-2.md` (format at the bottom). Branch:
> continue on `restore-corruption`.

## 0. Where the last session actually left us (corrected reading)

Established with evidence (do not re-derive):
- Phase-B/eagle-fallback: unrelated (corrupts either way).
- RAM-tier residency, transfer, and GPU placement are byte-faithful for
  ALL of: store→load (all 4 groups, 13,650 rows) and load→GPU-read
  (mamba groups 0–2, 144 rows) — **including for a corrupted restore**.
- The corrupted request's first forward step reads back *exactly the
  bytes that were stored*. Deterministic per conversation.
- Clean-vs-corrupt A/B (c0 vs c1, same run, 2s apart): load jobs are
  structurally identical — one job each, 73 keys (3 mamba + 70
  attention), correct key↔dst pairing and group ordering, same 58,240
  hit depth (verified this session from `qwen.log.phaseC-clean-run`
  lines 16432/17954). The load-job format is NOT the difference.

**Correction to the results file's elimination table:** the row "store
captured wrong content, Mamba groups 0-2 — ELIMINATED" is a logic error.
(a)=(b)=(c) proves the *pipeline* is faithful, not that the captured
content was *correct* — that's what the (never-obtained) block-70
reference value was for. This hypothesis is ALIVE and is now one of only
two left standing:

- **H-STORE**: the tier bytes are wrong from the moment of capture
  (poisoned at store time), then faithfully transported into token soup.
  Supporting pattern: c0 — the only conversation primed on an empty pool
  — is clean in every run; c1/c2/c3 — primed under eviction pressure,
  where a request's own superseded mamba checkpoints are freed and
  recycled within 1–2 steps of the store job that reads them — corrupt
  in every run. Also consistent with the local-vs-offload one-chunk gap:
  the 58,240 checkpoint is precisely a block local caching frees almost
  immediately (which is why local reuse resumes at 57,408 — it never
  keeps that checkpoint), so only offload restores ever *consume* it.
- **H-RESTORE**: the bytes land correctly but something clobbers or
  bypasses them before/at first use — possible only for the attention
  group (index 3), the one group with no GPU-read-side check. Candidate
  mechanism to keep in mind: post-load zeroing of "new" blocks
  (`new_block_ids_to_zero` — the same machinery implicated in the one
  historical AcceleratorError crash).

## 1. Phase A — the master fork (black-box, no code changes, ~30 min)

**A1. Restart-replay (run this first; it splits H-STORE vs H-RESTORE).**
1. Fresh fs tier dir; default production config; run
   `test_kv_cache_identity_corruption.py --convs 3 --rounds 2` and
   confirm the usual outcome (c1 corrupt, note the exact corrupted
   text). Archive the log.
2. **Restart the server** (RAM tier gone; fs tier persists;
   `PYTHONHASHSEED=0` makes the stored chunks reachable again). Confirm
   idle. Replay **only c1's turns** (the script builds deterministic
   prompts — reuse `build_prompt(1)` / extract the exact request bodies
   from `requests/*.log` of step 1 so content is byte-identical). Turn
   1's prefill should hit the fs tier (`fs -> RAM` promotions then
   `RAM -> VRAM`, confirm `GPU had 0 tok` or a large offloaded-hit line;
   if the first turn instead recomputes, send turn 2 as well and judge
   on whichever request actually restored).
3. Read the reply.

**Corrupt on the fresh, idle server → H-STORE confirmed** (the poison
travels with the tier bytes; no concurrency, new process, new GPU block
layout — everything restore-side is different, only the stored bytes are
the same). **Clean → H-RESTORE confirmed** (same bytes, different
runtime conditions → the stored bytes are fine).

Run A1 twice for confidence. This single experiment decides which half
of Phase C to execute.

**A2–A4: mechanism-class toggles** (run after A1, each is one repro run;
they narrow the *mechanism* within whichever branch A1 picked). For each,
change ONE thing in the launch config, rerun
`test_kv_cache_identity_corruption.py --convs 3 --rounds 2` (fresh fs
dir each time so runs don't cross-contaminate), record corrupt/clean
WITH exposure proof:
- **A2**: remove `--async-scheduling`.
- **A3**: remove the speculative config (no MTP drafting; also drop
  spec-dependent flags if startup complains).
- **A4**: `MAX_SEQS=1` (full serialization of priming and restores).

Interpretation grid (approximate, record surprises verbatim):
- A2 clean → async-scheduling interaction (optimistic bookkeeping /
  free-vs-copy or zero-vs-copy stream race).
- A3 clean → spec-decode interplay (plan 1 §3's optimistic storability,
  accepted-token bias, or spec-block state selection at capture).
- A4 clean → cross-request contention required → strongly H-STORE with
  a reuse race; A4 corrupt → structural, not load-dependent.

**A5 (only if A1 says H-STORE): serialized-priming test.** Prime c0,
restart, prime c1, restart, prime c2 (each priming runs uncontended on
an empty pool; fs tier accumulates all three). Then run the alternating
rounds normally. All-clean → store-time contention poison confirmed
independently of A1; still-corrupt → the pressure framing is wrong,
record and move on.

## 2. Phase B — forensics on the existing log (no GPU time; parallel with A)

On `qwen.log.phaseC-clean-run` (KEEP this file), trace the full store
history of c1's restored chunks vs c0's:
1. Identify c0/c1's round-0 request ids (the ~61s cold primings).
2. For the newest attention chunks and the mamba chunk at 58,240
   (offload key suffix `\x00\x00\x00\x0{0,1,2}` for mamba, `...\x03` for
   attention — the LOAD_JOB_BLOCKS lines at 16432/17954 give the exact
   key bytes to grep for): find each chunk's STORE job creation
   (`STORE_JOB_BLOCKS` / `Request ... offloading N chunks` lines), its
   src block ids, submission, and completion (`STORE_DONE` checksum
   lines carry job/block identity).
3. Between creation and completion, grep for reuse/free events touching
   those src block ids (`REALLOC`, eviction lines, `jobs_to_flush`,
   `MAMBA_FREE`, `new_block_ids_to_zero`). Build a per-chunk timeline
   table: created-at, src block, freed-at?, reallocated-at?, flushed?,
   completed-at.
4. Deliverable: the c0-vs-c1 differences in that table. If c1's src
   blocks were freed/reallocated mid-job and c0's never were, H-STORE
   has its mechanism; if the timelines are identical, that's equally
   important (records that the poison — if H-STORE — enters *before*
   job creation, i.e. the block already held wrong state when captured).

## 3. Phase C — targeted instrumentation (execute only the branch A1 picked)

### If H-STORE (tier bytes wrong at capture)

The question becomes: *what was in the block at copy time, vs what the
boundary state should have been?*

1. **Cross-run reference via CRCs** (no connector needed in the
   reference run): offload keys and the checksum instrument are already
   content-deterministic (`PYTHONHASHSEED=0`, temperature 0). Run the
   corrupting recipe once with `VLLM_KV_RESTORE_CHECKSUM=1` and record
   the (a) STORE_DONE crc for c1's mamba chunk at 58,240 (all 3 groups,
   per layer). Then run c1's prompt **alone, connector disabled**, with
   a small added hook that logs the same per-layer crc of the mamba
   state block the moment turn-1's own prefill completes the 58,240
   boundary (hook `preprocess_mamba`'s copy decision or the align
   postprocess — fire when a request's state at list position 69 is
   superseded, i.e. right after the boundary state is final; key the log
   line by `num_computed_tokens` so the boundary is unambiguous).
   Compare crc sets. Mismatch → captured-content wrong, and the diff is
   now measurable; match → the mamba chunk is *correct* and suspicion
   moves to the attention chunks' content (do the analogous attention
   comparison: store-time (a) crc per attention chunk vs a
   connector-disabled recompute hook that crcs each attention block as
   its last token is written — heavier; sample a handful of chunks, not
   all 70).
2. **Drift detector**: extend the checksum patch to also crc the GPU src
   bytes at store-job *creation* time (scheduler step, cheap GPU sum) and
   compare with the (a) crc at copy completion. Different → the block
   changed between the scheduler deciding "this chunk is final" and the
   copy reading it (free/realloc/zero race or late state write) — log
   which block ids drift and correlate with Phase B's timeline.
3. Fix will follow from which comparison breaks; candidates to have in
   mind (do NOT pre-implement): capture the checkpoint synchronously at
   job creation into a staging buffer; gate storability on verified (not
   scheduled) tokens; pin checkpoint blocks until their store completes
   (the flush-protect mechanism extended to cover pre-copy content
   stability, not just pre-reuse flushing).

### If H-RESTORE (bytes fine, clobbered/bypassed at use)

1. **Attention GPU-read checksum**: add the missing (c)-equivalent for
   group 3 — at the restored request's first scheduled step, crc each
   restored attention block (walk the request's group-3 block table row
   for the loaded range) and compare against (b). Do it in the runner
   right before the forward launch, gated by the same env var.
2. **Zeroing audit**: log every `new_block_ids_to_zero` (or whatever the
   zero-init path is named — find it via the AcceleratorError traceback
   in the docs) with block ids and timestamps; intersect with the load
   dst ids from `LOAD_JOB_BLOCKS`. Any overlap after load completion =
   the mechanism.
3. **First-read positioning**: if checksums match and nothing zeroes,
   instrument what the attention kernel actually consumes: dump the
   block table row the first forward uses vs the dst ids loaded —
   mismatch means the request was rescheduled/reshuffled between load
   and run (preemption, retry) and reads different blocks than were
   filled.

## 4. Phase D — fix + gates

Same gates as `RESTORE_CORRUPTION_PLAN.md` §9, unchanged: 3 consecutive
clean `test_kv_cache_identity_corruption.py` runs with proven exposure,
coherence-checking added to `test_offload_regression.sh`, eviction-variant
needle test, 15-min 3-way simulate run, `verify.sh` green, doc updated
(including correcting the plan-2 elimination-table row per §0 above),
temporary debug patches reverted from the venv.

Also fold in (as its own patch, after the main fix): plan 1 §3's
spec-boundary optimistic-storability defect if Phase A3/C confirmed it
matters, or file it as a documented known-issue with the
`STORE_AHEAD_OF_VERIFY` evidence if not.

## 5. Results file format (`RESULTS-restore-corruption-2.md`)

- **TL;DR**: which fork A1 picked (H-STORE / H-RESTORE), root cause if
  reached, fix status.
- **Phase A**: outcome of each toggle with exposure proof and verbatim
  replies; A1 twice.
- **Phase B**: the c0-vs-c1 per-chunk store timeline table, raw log
  lines for anything anomalous.
- **Phase C**: the crc comparison tables (store-time vs reference, or
  read-side vs (b)), raw lines.
- **Fix + gates**: as in plan 1.
- **Anomalies / open questions**: same rules — raw evidence over
  interpretation; if the root cause is still not found, the deliverable
  is the updated elimination table plus the single sharpest next
  experiment.
