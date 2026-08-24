# Plan 5: close Bug 2 (the unwritten chunk-0 store) and finish the investigation

> **For the executing agent (Opus/Sonnet):** self-contained follow-up to
> plans 1–4 (plan 1 §4 ground rules + plan 4's disk-hygiene rule apply)
> and `RESULTS-restore-corruption-4.md`. Write findings to
> `RESULTS-restore-corruption-5.md`. Branch `restore-corruption`.

## 0. The corrected picture (read before acting)

Session 4 left "chunk 0" as a suspect alongside an unexplained bisection
ambiguity. One fact closes the ambiguity — verified in code
(`get_sliding_window_size_in_chunks`, offloading/scheduler.py): **Mamba
groups load only their single newest chunk on restore
(`sliding_window_size_in_chunks == 1`); attention loads the whole chain
including chunk 0.** Therefore:

- Plan 3's mamba-only-scribble corruption was Bug 1 (reserved conv rows
  in the *loaded boundary chunk* — now fixed and scan-verified).
- Plan 3's attn-only-scribble corruption was Bug 2 (attention chunk 0,
  loaded by every full restore).
- Production mechanism: every full restore injects 832 tokens of
  stale/foreign K/V at positions 0–831; attention is global, so all
  later tokens attend into it → soup from the first decoded token,
  magnitude-dependent (virgin zeros benign — c0; mild scribble benign;
  stale real KV or loud scribble → collapse).
- Mamba chunk 0 being unwritten is *irrelevant to restores* (never
  loaded) but still worth fixing at the store side for hygiene — the
  tier shouldn't hold garbage under valid keys.

Also note: the `REALLOC blocks=[1..14]` timestamp correlation from
session 4 is probably benign (first-ever pool allocation; blocks
1/5/9/13 are just each group's position-0). Do not build on it without
the Phase B measurements.

## 1. Phase A — confirm the picture (3 cheap runs, no new code)

Each: scribbled isolated `c1` priming → restart unscribbled → replay
with confirmed full-restore exposure. Bug 1's fix stays applied.

- **A1: `SCRIBBLE_KV_ONLY=mamba`** — prediction: **CLEAN** now (Bug 1
  fixed; mamba chunk 0 never loaded). If corrupt: the picture is wrong —
  stop, scan the stored chunks, report.
- **A2: `SCRIBBLE_KV_ONLY=attn`** — prediction: **CORRUPT** (chunk 0).
- **A3: per-request or per-session?** One scribbled process, prime `c1`
  then `c2` sequentially (contended is fine — we're probing the store,
  not the restore). Stop server; run `scan_kv_chunks.py`. Question: does
  *each* conversation's chunk 0 survive scribble, or only the session's
  first-ever chunk? This determines production blast radius (every
  conversation's first 832 tokens vs. only the first request after boot)
  and discriminates mechanism (per-request store-timing bug vs.
  startup/first-step ordering bug).

## 2. Phase B — ground Bug 2's mechanism (measure, don't infer)

Extend the checksum debug patch (same env gate) with a chunk-0-focused
trace; all comparisons same-run (cross-run CRC remains invalid):

1. **At store-job creation** (scheduler, `_build_store_jobs`): log for
   every job: job id, chunk range, `req.num_computed_tokens`,
   `num_scheduled_tokens`, `num_tokens_after_batch` — i.e. whether the
   job covers tokens whose forward hasn't run yet (expected true for
   every chunk under the optimistic formula; record it anyway).
2. **At worker submission** (`submit_store`): log job id, wall time, and
   — the key measurement — a **GPU-side checksum of the src blocks
   taken on the transfer stream immediately after its
   `wait_stream(compute)`** (i.e. hash what the copy will actually see).
   Also record which CUDA stream `current_platform.current_stream()`
   returned at that moment (stream id), and whether the *producing*
   step's forward had been enqueued/completed — bracket each forward
   with recorded events in the runner (debug-gated) and `query()` them
   at submit.
3. **At store completion**: the existing CPU-side hash.
4. Run the scribbled priming once. For chunk 0 vs chunks 1–3, compare:
   - GPU-at-submit == CPU-at-completion? (should always match — the copy
     is the read; a mismatch means mid-copy mutation)
   - GPU-at-submit == scribble? → the blocks genuinely held garbage at
     submit → the wait/ordering failed or the forward hadn't produced
     yet → the stream/event data from (2) says which.
   - GPU-at-submit == real data but file == scribble → wrong pointers /
     wrong blocks — audit the src spec for job 0.
5. Deliverable: a one-paragraph mechanism statement for Bug 2 backed by
   these numbers, session-4-style.

## 3. Phase C — fix

**Preferred fix (also closes plan 1 §3's optimistic-storability defect
in the same move): make storability follow *completed* tokens, not
scheduled ones.** In `_build_store_jobs`, for non-finished requests,
derive `num_tokens_after_batch` from tokens whose forward output is
verified/complete (i.e. drop the `+ num_scheduled_tokens` optimism, or
use the verified count under async scheduling — Phase B's data will show
which counter is trustworthy at that point). Effect: every chunk's store
job is created one step later than today, when its content provably
exists; chunk 0 stops being special; decode/spec-boundary chunks stop
being storable on unverified tokens. Check the final-store path for
finished requests still covers the tail (it uses `req.num_tokens` — all
verified at finish — unchanged).

If Phase B instead shows a pure ordering/stream bug at first submission
(content existed, wait didn't cover it), fix that at the worker
(explicit event dependency on the producing step) — and *still* consider
the storability hardening as a second, separate patch, since plan 1 §3
remains real.

Add the hygiene half: mamba chunk 0's unwritten store (harmless to
restores, wrong in the tier) should disappear with the same fix —
verify via scan rather than assuming.

## 4. Phase D — gates (the full battery, at last)

1. **Star gate**: scribbled (0x7B, all groups) isolated prime → restore,
   spec ON: clean. Then the same with `ff` (NaN) pattern: clean.
2. **Scan gate**: `scan_kv_chunks.py` on a scribbled priming's tier:
   zero surviving runs anywhere except provably-dead padding (list any).
3. **Selector gates**: mamba-only and attn-only scribble recipes: clean.
4. `test_kv_cache_identity_corruption.py --convs 3 --rounds 2` and
   `--convs 4 --rounds 3`: 3 consecutive clean, exposure-proof runs.
5. Legacy gates (plan 1 §9): coherence-checking
   `test_offload_regression.sh`, eviction-variant needle test, 15-min
   3-way `simulate_hermes_traffic.py` (spot-check outputs, acceptance
   telemetry sane, no hangs/crashes), `verify.sh --no-server` green.
6. **Perf sanity**: confirm the storability change didn't measurably
   reduce offload coverage (stored-chunk counts and hit depths in the
   simulate run comparable to RESULTS-hit-diverged numbers).
7. Docs & closure: `docs/kv-offload-cache-identity-corruption.md` →
   FIXED with the full two-bug mechanism; closing addendum on the
   RESULTS chain; draft upstream-issue summaries (Bug 1: reserved conv
   rows never initialized; Bug 2: per Phase B; plus the full-disk
   stuck-RETRY robustness bug from session 3) for the user to decide on
   filing; revert debug patches from the venv (keep files, marked
   temporary); delete the remaining `vllm_kv.plan4-scribbled` stash and
   this session's superseded stashes; propose the branch-merge step
   (`restore-corruption` → `request_logging`) but do not merge without
   the user's go-ahead.

## 5. Results file format (`RESULTS-restore-corruption-5.md`)

- **TL;DR**: A1/A2/A3 verdicts; Bug 2 mechanism in ≤3 sentences; fix;
  full gate table (every gate, pass/fail, one line of evidence each).
- **Phase A**: outcomes with exposure proof; A3's per-request answer.
- **Phase B**: the chunk-0-vs-chunk-1..3 measurement table, raw lines.
- **Phase C**: patch name(s) + diff summary + why the fix point is
  sufficient (who else consumed the old behavior).
- **Anomalies/open questions**: as always. If any gate fails, stop at
  the first failure and report rather than proceeding.
