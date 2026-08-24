# Plan: root-cause the KV-offload restore corruption (mamba/GDN token soup)

> **For the executing agent (Opus/Sonnet):** this file is self-contained; read
> it fully before touching anything. Prior context lives in
> `docs/kv-offload-cache-identity-corruption.md` (the open bug + everything
> already ruled out — do NOT re-derive that list) and
> `docs/mamba-align-prefill-leak.md` (background on the offload stack; long,
> skim only if needed). Work on branch `request_logging`. Write findings into
> `RESULTS-restore-corruption.md` (format at the bottom) for the user to relay
> back for analysis. The goal of this plan is **root cause with evidence**,
> then a fix — not a workaround. Disabling offloading is already known to be
> clean and is not a result.

## 0. The bug in one paragraph

A conversation whose prefix is evicted from GPU and restored from the
offload tier (RAM and/or NVMe) decodes token soup from the very first
token, deterministically (4/4 in the repro), while a conversation that
stays GPU-resident is always clean. Reproduces on a pure RAM-tier restore
(zero `fs -> RAM` promotions), on a clean single-config cache, with the
cache-identity fix applied. Repro: `venv/bin/python
test_kv_cache_identity_corruption.py` (two ~58K conversations alternating
turns; the restored one corrupts). Only proven-clean config: offload
connector disabled.

## 1. What is already established — do not re-investigate

From `docs/kv-offload-cache-identity-corruption.md` (all verified with
dedicated tests): NOT the prefill threshold (832 vs 512), NOT the vision
tower, NOT the mamba-align free path (`MAMBA_FREE` fired 0 times), NOT
O_DIRECT alignment, NOT the disk-cap eviction, NOT the NVMe tier
specifically, NOT the mixed-config cache directory (real separate bug,
fixed in `patches/kv-offload-cache-identity-mamba-state.patch`).

Additionally verified by direct code reading this session (2026-08-24,
Fable) — sound, do not re-chase without new evidence:

- **Store copies are correctly ordered after compute.**
  `v1/kv_offload/cpu/gpu_worker.py`: the transfer stream does
  `wait_stream(compute)` for GPU→CPU, stores are deferred to
  next-step-begin (`prepare_store_kv` → `_unsubmitted_store_jobs` →
  `start_kv_transfers`), so the copy sees the completed step's content.
- **Loads are host-gated**: load completion is polled via CUDA events
  (`get_finished`) and the scheduler only runs the request afterward.
- **The index chain is self-consistent**: store associates chunk `k`
  with "state after (k+1)*832" (`MambaManager` convention: "Block i ends
  at token (i+1)*block_size"); a full restore loads exactly the chunk
  ending at the resume boundary into block-list position `k`
  (`update_state_after_alloc` per-group scan); `preprocess_mamba`
  (`v1/worker/mamba_utils.py`) reads the resume state from position
  `(num_computed_tokens - 1) // 832` — the same position. No off-by-one
  at this level.
- **The uniform 28,966,912-byte tier file size is innocent**: it equals
  17 attention layers × 1,703,936 B (832 tok × 2(K,V) × 4 kv-heads ×
  256 head_dim × 1 B fp8), and this hybrid model's mamba state shares
  the same unified per-block page geometry, so every block weighs the
  same. Not a layout smoking gun by itself.

## 2. The two load-bearing gaps this plan exists to close

**Gap 1 — "restore used to work" was never actually verified.** Every
pre-2026-08-23 "restore works" datapoint was a latency/hit-line
measurement. `needle_recall_test.py` turn 2 was a GPU-local hit (solo
conversation — nothing evicted it), and `test_offload_regression.sh`
asserts timing + `hit N offloaded tokens` lines, never output coherence.
Overnight load was `simulate_hermes_traffic.py`, which never reads
replies. So the corruption may have existed since mamba restores first
started functioning (2026-08-22) — or not. This must be settled.

**Gap 2 — the one config delta inside the corruption window was never
reverted as a test.** `VLLM_OFFLOAD_EAGLE_FALLBACK=0` (Phase B of the
HIT_DIVERGED work) went live 08-23 ~08:18; first observed garbage 08-23
12:10. Phase B changed exactly **which chunk a full restore resumes
from**: before it, the eagle pop in the connector's `_lookup()` clamped
every full restore one chunk short of the newest stored chunk; after it,
restores resume at the newest one. If the newest chunk stored per turn is
systematically bad, pre-Phase-B restores were *accidentally* clean.

## 3. Known concrete defect (fix in Phase F regardless of the repro's cause)

**Optimistic storability across spec-decode boundaries.**
`_build_store_jobs()` (`distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`)
computes `num_tokens_after_batch = req.num_computed_tokens +
num_scheduled_tokens` — which includes the current step's
still-unverified speculative tokens (MTP, 3 spec tokens here). A mamba
chunk whose 832-boundary is crossed only by scheduled-but-later-rejected
tokens becomes "storable" while its block still holds pre-boundary
running state; the store fires, `next_stored_chunk_idx` advances (never
re-stored), and the wrong bytes sit permanently in a content-addressed
tier under that boundary's key. This poisons boundaries crossed during
*decode*. NOTE: it does **not** explain the current repro — there, the
restored chunk's boundary (58,240 = 70×832 < 58,669-token prompt) was
crossed during plain aligned *prefill*. Treat it as a second bug.

## 4. Ground rules

- **Branch** `request_logging`; commit logically-separate changes
  separately; never commit to `main`.
- **Patch workflow** (matches every existing patch): edit the live file
  under `$SP` = `venv/lib/python3.12/site-packages/vllm/`, then write a
  unified diff with `a/`,`b/` paths relative to `$SP` into
  `patches/<name>.patch` with a prose header (root cause, evidence, fix,
  verification — see `patches/offload-eagle-misclassification-mamba.patch`
  for style). `bash verify.sh --no-server` must show every patch applied.
  If changing behavior an existing patch owns, regenerate that patch, do
  not stack a conflicting one.
- **Locate code by symbol/grep, never by line numbers quoted here**
  (applied debug patches shift lines).
- **Server**: `bash micke-start.sh`; log `qwen.log`; health
  `curl -sf 127.0.0.1:18020/health`. Before restarting, check
  `ls -lt requests/ | head` — files modified in the last ~10 min mean
  live user traffic; if in doubt, stop and ask.
- **Preserve evidence before generating more.** Before the first
  restart: copy the current `qwen.log` (if present) and any log the doc
  references to `qwen.log.<date>-<purpose>` files. Never delete or clear
  `/d/nvme_cache/vllm_kv` or the RAM-tier contents unless a step says to
  — and when a clean cache is needed, move the directory aside instead of
  deleting it.
- **Determinism**: all repro/validation generation at temperature 0.
  `test_kv_cache_identity_corruption.py` is the gate — it reads the
  actual replies. Run it with the same settings each time; record exit
  code AND the verbatim first-line of each response.
- **A "pass" only counts with proven exposure**: check the run actually
  restored (`GPU had 0 tok` lines / 5s+ turns). The doc's "Caution about
  clean runs" section is binding: low-exposure passes are not evidence.
- **If the engine hangs** (frozen `GPU KV cache usage`,
  `VLLM::EngineCore` at 100% CPU state R): `kill -USR1 <EngineCore pid>`,
  save `/tmp/vllm_faulthandler_dump.txt`, record it, restart.
- **Out of scope**: do not build the synchronous store-on-eviction
  coupling; do not disable offloading and call it done; do not touch
  `patches/mamba-align-stale-state-queue.patch`; do not weaken the
  HIT_DIVERGED reconcile safety semantics.
- `patches/offload-verbose-eviction-debug-logging.patch` and
  `patches/kv-transfer-request-logging.patch` stay applied — they are the
  measurement instruments (`request-logging offload-verbose:` and
  `KV LOAD/STORE` lines).

## 5. Phase A — the Phase-B revert test (cheapest, biggest split; ~15 min)

1. In `micke-start.sh`, comment out `VLLM_OFFLOAD_EAGLE_FALLBACK=0`
   (re-enabling the connector's eagle mark-everything fallback: startup
   log should again say `EAGLE/MTP draft attention groups [3] detected`).
2. Move the fs cache dir aside (fresh dir) so stale chunks from the
   corrupting era can't confound; restart the server.
3. Run `venv/bin/python test_kv_cache_identity_corruption.py`. Confirm
   real exposure (full restores happened). Run it twice.
4. Restore `micke-start.sh` afterward (git checkout of that hunk) unless
   Phase E decides otherwise.

**Decision:**
- **Clean (2/2, with exposure)** → the defect is in the *newest stored
  chunk per turn* that only post-Phase-B restores consume. Focus Phase C
  on the final chunks' store content/timing (including §3's optimistic
  storability, and what the final prefill/decode steps leave in the
  newest mamba block at store-copy time).
- **Corrupt** → Phase B is exonerated; this is a latent day-one restore
  bug. Phase C runs with full scope.

Record the result either way with log excerpts (the `KV LOAD` lines +
the corrupted/clean replies).

## 6. Phase B — settle history from existing logs (no GPU time)

Search 08-22-era artifacts for turns that followed a genuine full restore
and read the *logged replies*:

1. `requests/*.log` from 2026-08-22 (the request-logging patch captures
   bodies; check whether responses are captured too — if only requests,
   a turn N+1 body *contains* turn N's assistant reply, which is exactly
   what we need: find conversations whose turn N reply followed a
   restore and read that reply text inside turn N+1's messages array).
2. Correlate with that day's `qwen.log*` files (`hit N offloaded tokens
   after 0 GPU hit tokens` = full restore; the archived
   `qwen.log.evidence-mamba-free-never-fired.log` is 08-23 but check for
   other archived logs).
3. Verdict: were any pre-Phase-B post-restore replies demonstrably
   coherent (or demonstrably soup)? Quote them.

This cross-checks Phase A: A-clean + B-finds-coherent-08-22-restores is a
consistent picture; A-corrupt + B-finds-coherent-restores would mean the
trigger is something else that changed 08-22→08-23 and needs a harder
look (report, don't guess).

## 7. Phase C — checksum triangulation (the decisive instrument)

Build one temporary diagnostic patch (`patches/restore-corruption-checksum-
debug.patch`, clearly marked temporary) that logs a content checksum per
(offload key, group) at three points:

- **(a) store**: after the GPU→CPU copy completes for a store job, hash
  the CPU-side bytes of each chunk (per key). CPU-side hashing avoids
  GPU syncs; do it where store completion is already processed
  (`get_finished` / transfer completion path), reading from the CPU
  offload region via the job's dst spec. Cap overhead: hash only when a
  debug env var is set, e.g. `VLLM_KV_RESTORE_CHECKSUM=1`, and consider
  hashing only the first+last 64KB plus length if full-page hashing is
  too slow (state divergence will show there too; note which variant you
  used).
- **(b) load**: at load submission, hash the same CPU-side source bytes.
  (a)==(b) for the same key proves RAM-tier integrity across the
  residency window.
- **(c) GPU truth**: for the *restored* request's first step, in
  `preprocess_mamba` (align path), when `prev_state_idx` comes from
  `(num_computed_tokens - 1) // block_size` (i.e. a resumed request),
  log: req_id, num_computed_tokens, prev_state_idx, the physical
  block_id at that position per mamba group, and a GPU-side checksum
  (`tensor.view(torch.int64).sum()` of that block's state pages is
  enough; a `.item()` sync here is acceptable under the debug env var).
  Also log the same checksum for a *locally-resumed* request (resident
  conversation's next turn) as the healthy baseline.
- **(reference) recompute**: run the corrupting conversation once with
  the connector disabled and dump the same GPU-side checksum at the same
  boundary (the (c) hook fires for local resumes too, giving you the
  ground-truth state checksum for that exact prefix at that exact
  boundary — same tokens, temperature 0, so state bytes should be
  reproducible; verify reproducibility by running it twice before
  trusting it).

Then run the repro with checksums on and classify:

| observation | meaning | next |
|---|---|---|
| (a) ≠ (b) for a restored chunk | RAM-tier byte corruption in residency | audit CPU region alloc/reuse/eviction for that key's lifetime |
| (a) == (b), GPU-read (c) ≠ (b) | load transfer or placement wrong | audit load dst mapping (`update_state_after_alloc` dst_block_ids vs the block preprocess reads; worker `compute_sub_block_ptrs` for mamba refs) |
| (a) == (b) == (c) but ≠ reference | **store captured wrong content** | audit what the newest block held at store-copy time: §3's optimistic storability; ordering of the align postprocess vs store submission within the crossing step; whether the stored block was the CoW/partial-tail block (`_producer_partial_tail_reqs` machinery) vs the in-list block |
| all equal incl. reference | state bytes are right; corruption is elsewhere | instrument the first forward step after restore: attention-group restore checksums next (same (a)/(b)/(c) treatment for group 3), then `num_accepted_tokens`/token_bias on the restored request's first `preprocess_mamba` call |

Hash **all four groups**, not just mamba — if attention chunks are also
wrong, the mamba framing is a red herring.

## 8. Phase D — the spec-boundary optimistic-store check (§3)

Add (fold into the same temporary debug patch) one log line at mamba
store-job build when the chunk's end boundary exceeds
`req.num_computed_tokens` at build time (i.e. content depends on the
current step's outcome):
`STORE_AHEAD_OF_VERIFY req=... group=... chunk=... boundary=... computed=... scheduled=...`.
Run a decode-heavy workload (a conversation generating 2-3K tokens so
replies cross several 832 boundaries), then grep. Any hits during decode
with subsequent spec-rejection in the same window = confirmed second bug.
Report frequency; the fix (Phase F) caps storability at verified tokens
(e.g. exclude the current step's unverified tokens from
`num_tokens_after_batch` for non-finished requests, or defer the chunk
one step) — but only implement after root-causing the main repro, and as
its own patch.

## 9. Phase E — fix + gates

Implement the fix for whatever Phase A–C converged on, as a properly
documented patch. Then ALL of:

1. `test_kv_cache_identity_corruption.py` — 3 consecutive clean runs
   with confirmed exposure (`GPU had 0 tok` present), default config
   (Phase-B env var restored to whatever Phase E decides is correct —
   state the decision explicitly).
2. `test_offload_regression.sh` — still passes (timing + hit lines), AND
   extend it to also grep conv A turn 2's reply for coherence (fail on
   `!!!!` runs / replace its pass criterion with a scripted check similar
   to `test_kv_cache_identity_corruption.py`'s classifier).
3. `needle_recall_test.py` variant with real eviction: turn 1, then a
   second large conversation to evict, then the needle question — the
   recall must survive an actual tier restore. (The existing solo
   version stays as the local-hit check.)
4. `simulate_hermes_traffic.py --threads 3` ~15 min: zero corrupted-
   looking replies (spot-check outputs), no hangs, no `AcceleratorError`,
   preemptions/HIT_DIVERGED counts comparable to the RESULTS-hit-diverged
   baseline.
5. `bash verify.sh --no-server` fully green (modulo the 4 pre-existing
   DFlash2-family failures noted in RESULTS-hit-diverged.md).
6. Revert the temporary checksum/debug patch from the venv (keep the
   patch file in `patches/` marked temporary, or delete it — state
   which), and update `docs/kv-offload-cache-identity-corruption.md`:
   status OPEN → FIXED, with the mechanism, in the doc's
   append-and-correct style.

## 10. Results file format (`RESULTS-restore-corruption.md`)

- **TL;DR**: root cause in ≤3 sentences; fix; current status of the
  Phase-B env var and why.
- **Phase A**: both run outcomes with exposure proof + verbatim replies.
- **Phase B**: what the 08-22 logs showed, with quoted reply excerpts
  and the hit-lines they correlate to.
- **Phase C**: the checksum table for at least one corrupted restore
  (all four groups, all three/four measurement points), and the healthy
  local-resume baseline row. Raw log lines, not summaries.
- **Phase D**: STORE_AHEAD_OF_VERIFY hit count + a sample line, or "zero
  hits over N decode-boundary crossings".
- **Fix**: patch name, mechanism, and each Phase-E gate's result.
- **Anomalies & open questions**: anything unexpected, each with what
  was tried and raw evidence. If the root cause was NOT found, this
  section is the deliverable: the full elimination table with evidence,
  and the sharpest remaining hypothesis with the experiment that would
  decide it.
