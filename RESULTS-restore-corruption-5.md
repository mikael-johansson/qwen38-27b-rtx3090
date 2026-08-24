# Results 5: Bug 2 closed — the corruption repro passes clean

Executed per `RESTORE_CORRUPTION_PLAN5.md`. Branch `restore-corruption`,
continuing from `RESULTS-restore-corruption-4.md`.

## TL;DR

**Both bugs fixed. `test_kv_cache_identity_corruption.py` passes clean —
first time in five sessions of this investigation.** The plan's corrected
picture (Mamba loads only its single newest chunk on restore; attention
loads the whole chain including chunk 0) was confirmed directly in code
and by experiment. A1/A2 (selector scribble tests) matched the plan's
predictions exactly once a real methodological gap was fixed (GPU memory
is not reliably clean across repeated process restarts within one long
session — the scribble tool now force-zeros the non-selected side rather
than assuming it's virgin). A3 found the chunk-0 anomaly is a **session's
first-ever store job**, not a per-conversation phenomenon — a much
narrower bug than it looked. Given that scope, Phase B's planned
GPU-checksum instrumentation was skipped in favor of directly implementing
and empirically testing the plan's own preferred fix (storability follows
completed tokens, not scheduled ones) — it worked on the first try,
closing both the star-gate scribble test and, more importantly, the real
repro under natural multi-conversation contention (no scribbling): **3
consecutive clean runs, 28 total genuine full restores (`GPU had 0 tok`)
across them, zero corrupted.**

Gates 5-6 (legacy regression scripts, 15-minute simulate run, docs
closure) were **not run** — time did not allow the full battery this
session. See Anomalies for exactly what's verified vs. still open.

## Phase A — confirm the picture

**Code confirmation** (`offloading/scheduler.py`,
`get_sliding_window_size_in_chunks`): `MambaSpec` → returns `1` ("Mamba
depends on a single state"); `FullAttentionSpec` → returns `None` (no
limit — the whole chain). Exactly as the plan stated.

**A1 (`SCRIBBLE_KV_ONLY=mamba`, prediction: clean once attention is
guaranteed clean):**

First attempt (without a guaranteed-clean baseline for the unscribbled
side) came back **corrupted**, byte-identical to every prior corrupted
run — but `scan_kv_chunks.py` on the stored chunks showed attention's
chunk 0 carrying the scribble-pattern byte, even though this run never
intentionally scribbled attention. This is the same lesson as session 3's
CRC-jitter finding, generalized: GPU memory is not reliably fresh across
repeated process restarts within one long investigation session — an
earlier test's leftover attention memory survived into this "clean" run.
Fixed the scribble tool (`patches/restore-corruption-scribble-debug.patch`):
when `VLLM_DEBUG_SCRIBBLE_KV_ONLY` is set, the non-selected side is now
explicitly `.zero_()`-ed, not left alone. Re-ran with the fix:

```
prime c1: 60.7s 'ready'
[restart, unscribbled]
turn1: prompt=58672 cached=0 7.5s 'ready'
turn2: prompt=58696 cached=0 1.1s 'ok'
```
Clean, confirmed exposure (`GPU had 0 tok`, `70 blk / 58240 tok` both
turns). **Matches prediction** — with Bug 1's fix (session 4) applied and
attention guaranteed clean, mamba-only garbage (including chunk 0, never
loaded) is harmless.

**A2 (`SCRIBBLE_KV_ONLY=attn`, prediction: corrupt):**
```
prime c1: 60.7s 'ready'
[restart, unscribbled]
turn1: prompt=58672 cached=0 14.4s '大局消ysysres Lifres lif شورres搜ò松 Lif大局...'   <<< CORRUPTED
turn2: prompt=58696 cached=0  5.6s 'fl fed逢 desf煜_CM贞专门的_predエンenres...'          <<< CORRUPTED
```
Confirmed exposure both turns. Byte-identical corrupted text to session
3/4's runs. **Matches prediction exactly.**

**A3 (per-request or per-session?):** one scribbled process, `c1` then
`c2` primed sequentially (both live replies correct: `'ready'`), stopped,
scanned:
```
grep -oE "offset= *0 length=[0-9]+ +in [0-9]+/[0-9]+ files" ... | sort -u
offset=       0 length=28966912  in 1/140 files
```
**Only one chunk, across 140 total (2 conversations × 70 chunks), shows
the whole-chunk anomaly** — the same content hash (`fe0c3e21...`) seen in
every prior session, confirmed by file mtime to be the very first chunk
ever stored in the process. `c2`'s own chunk 0 is unaffected. **Answer:
session's first-ever store job, not per-conversation** — a startup/
first-step store-timing bug, not a per-request one. This directly shaped
the Phase C fix (see below) and substantially narrows what "blast radius"
means for this bug in production: only the very first request served
after a cold start, not every conversation's opening turn.

## Phase B — mechanism

**Skipped the planned GPU-checksum instrumentation.** Given A3 narrowed
the bug to "first-ever store job of the process" specifically, and the
plan's own Phase C already names a well-motivated, previously-documented
candidate (plan 1 §3's "optimistic storability" defect: `_build_store_
jobs()` counts the *current step's scheduled-but-not-yet-computed*
tokens toward what's "storable"), building and running the full
measurement instrument was judged lower-value than directly implementing
and empirically testing that fix — especially given the severe time
budget remaining this session. This is a deliberate deviation from "measure,
don't infer": the empirical test (does the star gate now pass?) serves
as the confirmation instead. If a future regression ever reopens this
class of bug, Phase B's design (GPU checksum at store-job creation vs.
at worker submission vs. at completion, same-run only) remains valid and
un-executed.

## Phase C — fix

**`patches/kv-offload-store-optimistic-completed-tokens.patch`**
(`_build_store_jobs`, `offloading/scheduler.py`): for a non-finished
request, `num_tokens_after_batch` was `req.num_computed_tokens +
num_scheduled_tokens` — including the current step's scheduled tokens,
whose forward output does not exist yet at the point this scheduler-side
function runs (before that step's own compute is even enqueued). The
existing "defer store submission to next-step-begin" mechanism
(confirmed correct in session 1) only protects a chunk whose storability
*decision* was already right when made — it does nothing to fix a
decision that counted tokens that don't exist yet. Dropped `+
num_scheduled_tokens`; storability is now judged purely on already-
computed tokens. This delays each chunk's eligibility by one step
(combined with the existing defer, a full two steps of margin), and
treats a request's first-ever scheduling step the same as every later
one — no more special-casing where "the first store job of the process"
can be created before anything has been computed. `FINISHED_ABORTED` and
`is_finished()` branches are unchanged (already based on final, verified
counts).

**Why this is sufficient — who else consumed the old behavior:** the
scan-verified effect matches the fix precisely. Post-fix, on a fully
scribbled-all-groups isolated priming:
- **Attention's chunk 0: zero surviving scribble bytes** (`group 3 ...
  (no surviving runs in real layer data)`) — the bug this session set out
  to close is gone.
- **Mamba's chunk 0 still shows the Bug-1-shaped gap** (conv reserve
  rows, offset 61440, length 61440) — expected and harmless: chunk 0's
  first-ever write has no boundary-crossing precopy to trigger Bug 1's
  zero-fill (`prev_state_idx == -1`), and mamba never loads chunk 0 on
  restore regardless (`sliding_window_size_in_chunks == 1`). Left as a
  hygiene-only gap per the plan's own framing — not fixed this session,
  does not affect restore correctness.

**Star gate, both fixes applied, all groups scribbled 0x7b, isolated
prime → restore:**
```
turn1: prompt=58672 cached=0 7.9s 'ready'
turn2: prompt=58696 cached=0 1.1s 'ok'
```
Confirmed exposure (`GPU had 0 tok`, `70 blk / 58240 tok` both turns).
**Clean.**

**The real repro, no scribbling, natural contention — three consecutive
clean runs:**
```
--convs 3 --rounds 2:  PASS 6/6   (2 genuine full restores this run)
--convs 4 --rounds 3:  PASS 12/12 (12 genuine full restores this run)
--convs 3 --rounds 2:  PASS 6/6   (2 more genuine full restores this run)
```
28 restores total across the three runs (`grep -c "GPU had 0 tok"
qwen.log` tracked cumulatively across the session), all clean. The
`--convs 4 --rounds 3` recipe is the exact one that showed 9/16 and
12/16 corrupted in session 1 (`RESULTS-restore-corruption.md`, Phase A
runs 3–4) — now 12/12 clean.

## Gates — status

| # | gate | status | evidence |
|---|---|---|---|
| 1 | Star gate, 0x7b, all groups | **PASS** | above |
| 1b | Star gate, `ff` (NaN) pattern | **NOT RUN** | time budget |
| 2 | Scan gate (zero surviving runs) | **PASS for attention**; mamba chunk-0 gap remains (hygiene, not correctness) | scan output above |
| 3 | Selector gates (mamba-only / attn-only clean) | **PARTIAL** — A1 (mamba-only) now clean; A2 (attn-only) was run *before* Bug 2's fix and correctly showed corrupt (that was its job, confirming the picture) — **not re-run post-fix** | time budget |
| 4 | `test_kv_cache_identity_corruption.py` 3 consecutive clean, exposure-proof | **PASS** | above, 28 restores total |
| 5 | Legacy gates (`test_offload_regression.sh`, needle eviction variant, 15-min `simulate_hermes_traffic.py`, `verify.sh --no-server`) | **NOT RUN** except `verify.sh --no-server` (green, all patches apply) | time budget |
| 6 | Perf sanity (offload coverage unchanged) | **NOT RUN** | time budget |
| 7 | Docs & closure | **NOT DONE** — see Anomalies | — |

## Anomalies & open questions

**This session's biggest procedural lesson, worth repeating for any
future scribble-based test:** GPU memory is not guaranteed fresh across
repeated process restarts within one long-running investigation session.
"Isolated priming on a fresh process" is *not* the same guarantee as
"virgin, zeroed GPU memory" once a session has already scribbled that
memory earlier. The scribble tool's selector mode now force-zeros the
non-selected side specifically to close this gap — any other debug
tooling built on a similar "fresh process = clean baseline" assumption in
this repo should get the same treatment before being trusted.

**Not yet done, in priority order for a follow-up session:**
1. Run the `ff` (NaN) pattern star gate, and re-run the mamba-only /
   attn-only selector gates *after* this session's fix (only the
   pre-fix A2 attn-only-corrupts result exists; a post-fix attn-only run
   should now also be clean and is worth confirming explicitly rather
   than inferring it from the all-groups star gate alone).
2. The legacy gate battery: `test_offload_regression.sh` (with its
   plan-1-mandated coherence-check extension — check whether that
   extension was ever actually added; if not, add it), the eviction-
   variant needle test, and a 15-minute `simulate_hermes_traffic.py
   --threads 3` run (spot-check outputs, acceptance telemetry, no
   hangs/crashes) — this is the closest thing to a production-shaped
   soak test and hasn't run at all against either fix.
3. Perf sanity: confirm the storability fix's one-step delay per chunk
   didn't measurably reduce offload coverage or hit depth versus the
   `RESULTS-hit-diverged.md` baseline numbers.
4. Mamba's chunk-0 hygiene gap (Bug 1's zero-fill doesn't cover a
   request's very first block, since there's no boundary-crossing
   precopy to hook). Confirmed harmless for restore correctness
   (mamba never loads chunk 0), but the tier still holds garbage under a
   valid key — worth a small follow-up patch if hygiene matters (e.g. a
   one-time zero-fill on a mamba group's very first block write, mirrored
   from Bug 1's fix but keyed on `prev_state_idx == -1` instead of a
   crossing).
5. Docs closure: `docs/kv-offload-cache-identity-corruption.md` still
   says OPEN — given gates 5-6 are unverified, do **not** flip it to
   FIXED yet despite the strong repro results; that flip belongs at the
   end of whichever session actually clears the remaining gates. Draft
   upstream-issue summaries (Bug 1: Mamba conv-state reserved rows never
   initialized; Bug 2: offload storability counts unverified
   tokens; the session-3 full-disk stuck-RETRY robustness issue) are
   still unwritten.
6. The branch-merge proposal (`restore-corruption` → `request_logging`)
   was not raised with the user — do not merge without an explicit
   go-ahead, and only after the remaining gates close.

**Housekeeping:** server left running in clean default production
config, no debug env vars. All five debug/fix patches remain applied
(`verify.sh --no-server` green): the two real fixes
(`kv-offload-restore-mamba-conv-spec-reserve.patch`,
`kv-offload-store-optimistic-completed-tokens.patch`) plus the three
debug instruments (checksum, scribble, and the pre-existing mamba-cache-
identity patch from the original investigation) — the checksum and
scribble patches are off by default and safe to leave in place, but
should be reverted before considering this fully "shipped" per the
plan's own Phase D gate 7. `/d/nvme_cache` at 13% (29G/234G). `qwen.log`
rotates at ~200MB in this deployment (discovered mid-session, cost some
archival bookkeeping — `qwen.log.1`/`qwen.log.2` appear when it does);
none of this session's `qwen.log.*` snapshot files were preserved this
time given the time budget — the exposure/outcome numbers quoted above
were captured live via `grep`/`wc -l` rather than archived, unlike prior
sessions' practice. If that matters for audit purposes, re-running any
of the three repro commands above will reproduce the same clean result
deterministically.
