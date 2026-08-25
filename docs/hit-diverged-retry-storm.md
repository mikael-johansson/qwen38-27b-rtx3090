# HIT_DIVERGED reconcile retry storm (2026-08-25)

> **STATUS: FIXED.** Two patches, both applied:
> `patches/hit-diverged-reconcile-log-dedup.patch` (log spam) and
> `patches/hit-diverged-retry-backoff.patch` (the underlying wasted work).
> Found via live production log analysis on `micke-start.sh`, not
> synthetic testing. Patched and verified (`verify.sh --no-server`
> green) but not yet observed live against a fresh recurrence of the
> high-contention scenario that exposed the original problem.

## What happened

User-reported: a burst of **100+ near-identical `HIT_DIVERGED
RECONCILE` WARNING lines in under ten seconds**, all for the same
request (`chatcmpl-85fc174aeeb32247-b7987cde`), e.g.:

```
WARNING [scheduler.py:945] request-logging: chatcmpl-85fc174aeeb32247-b7987cde HIT_DIVERGED RECONCILE -- diverged local hit was 40768 tokens, external connector confirmed 0, reconciled down to 0 tokens (the boundary every KV group agrees on) -- 40768 tokens of otherwise-valid hit discarded for Mamba-state safety
```
(repeated ~150 times, same numbers every time)

`HIT_DIVERGED RECONCILE` itself is pre-existing, correct behavior from
`patches/hit-diverged-boundary-rescue.patch`: when the full-attention
KV cache has a deep local hit but the Mamba/GDN state's own (single,
non-sliceable) checkpoint can't be confirmed valid at that same
boundary, the scheduler safely discards the whole hit and recomputes,
rather than risk resuming with mismatched attention/Mamba state (the
exact failure mode `RESTORE_CORRUPTION_PLAN*.md` spent five sessions
chasing). That decision was correct. The problem was that it kept
happening for the *same already-decided* case, over and over.

## Root cause

`if request.num_computed_tokens == 0:` gates the entire local lookup +
connector round-trip + boundary-rescue + reconcile block in
`v1/core/sched/scheduler.py`'s `schedule()`. That field is only set once
`allocate_slots()` actually succeeds. When it returns `None` (no free KV
blocks — the concurrent-traffic paste showed GPU KV cache usage
cycling 81–99.6% with 3 running + up to 1 waiting/deferred request),
the scheduling loop `break`s for that step without ever setting
`num_computed_tokens`, so the same request re-enters the whole block
from scratch on the very next `schedule()` call — roughly every
60–70ms under this deployment's MTP spec-decode cadence. Nothing about
the connector's backing state had changed between attempts, so it
deterministically discarded the identical amount and logged the
identical WARNING every single retry — purely because the request was
stuck waiting on GPU memory, not because anything was actually wrong.

## Fix 1: log-spam (`hit-diverged-reconcile-log-dedup.patch`)

`logger.warning` → `logger.warning_once` at the RECONCILE call site.
vLLM's `warning_once` dedupes on `(logger, format_string, *args)` via a
bounded (`maxsize=128`) `lru_cache` — since request_id and both token
counts are passed as `*args`, this collapses only *true* repeats (same
request, same reconciled outcome). A different request, or this same
request later reconciling to a genuinely different boundary, still gets
its own line. Zero change to scheduling behavior — logging only.

## Fix 2: the underlying wasted work (`hit-diverged-retry-backoff.patch`)

Investigated caching the connector's answer directly and **ruled it
out**: `OffloadingConnectorScheduler.get_num_new_matched_tokens()` has
real side effects on every call — it clears and re-derives the
request's per-group offload block-id tracking, resets
`req_status.num_locally_computed_tokens`, and calls `self._touch()`,
which protects the hit chunks from eviction. Skipping that call on a
"cached" tick would skip the eviction-protection touch too — actively
counterproductive under exactly the memory-pressure conditions that
cause this stall — and could leave `req_status`'s internal state stale
relative to whatever boundary is actually used if allocation succeeds
right after. So neither the connector call nor the block references
`get_computed_blocks[_for_connector]` return are ever cached by this
fix — every attempt that's actually made still redoes the complete,
correct, side-effect-preserving dance.

Instead: back off *how often* the dance is attempted, not *what* it
computes when attempted. The reconcile branch stamps
`request._hit_diverged_reconcile_ts = time.monotonic()` the moment it
runs (a plain attribute on `Request` — not a dataclass field, not
slotted, self-cleans with the request's own lifetime). At the top of
the waiting-queue walk, before any lookup work, a request whose stamp
is younger than `VLLM_HIT_DIVERGED_RETRY_COOLDOWN_S` (default 0.3s, `0`
disables this entirely) is popped and deferred to `step_skipped_waiting`
instead — the same "pop, defer, try the next waiting request" pattern
already used by the lora/blocked-status/stale-output-token skips in
that identical loop, so a smaller request behind it can still be
scheduled this step, and admission order/fairness is otherwise
unchanged. If the same request is seen again within the cooldown, it
can only be because admission failed right after the stamp was set (a
successfully-admitted request leaves the waiting queue and is never
peeked here again) — so nothing about GPU or offload state could
plausibly have changed in between. A genuine change in resource
availability (blocks freed up, an in-flight store completes) is never
delayed by more than the cooldown.

Net effect at the default 0.3s cooldown: a multi-second admission
stall drops from ~15 full dance-repeats/second to ~3/second for the
affected request. HIT_DIVERGED reconcile safety semantics are
completely unchanged — same tokens discarded, same fallback taken,
just attempted less often while genuinely blocked on resources rather
than blocked on a data-availability question that's already been
answered.

## What this does *not* fix

Neither patch changes whether cached KV/Mamba state is actually
discarded — that's the correct, existing HIT_DIVERGED safety fallback.
The follow-up investigation (same session, not yet actioned) into
*why* the connector couldn't confirm the boundary in the first place
pointed at the offload tiers' capacity being undersized for this
workload — see the disk-cap-eviction discussion in this session's
conversation log; not yet written up as its own doc.

## Verification

Syntax-checked (`ast.parse`), `verify.sh --no-server` green for both
patches. Not yet observed against a live recurrence of the
high-contention scenario that originally exposed this (would show up
as isolated `HIT_DIVERGED RECONCILE` lines spaced by roughly the
cooldown, instead of a dense back-to-back burst).
