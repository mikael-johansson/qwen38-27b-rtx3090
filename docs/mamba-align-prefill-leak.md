# Mamba-align prefill memory leak (self-preemption on large single prompts)

> **STATUS as of 2026-08-22, ~15:30 UTC: FIXED. Both patches applied and
> verified at the default `--max-num-batched-tokens 2048`:**
> `patches/mamba-align-stale-state-queue.patch` (the leak fix) **and**
> `patches/offload-eagle-misclassification-mamba.patch` (fixes the KV-
> offloading regression the leak fix exposed -- a real, separate bug in
> the connector, not in the leak fix itself). See "The fix breaks KV
> offloading at 2048 -- root cause found and fixed" below for the full
> story, including everything that was ruled out along the way (kept for
> the record so it isn't re-derived). Verified: `test_offload_regression.sh`
> passes reliably, 2x concurrent ~88K-token prompts complete with zero
> preemptions, and a 3x concurrent ~120K-token stress test (360K tokens
> of combined demand against a ~172K-token pool) completes with zero
> preemptions and zero discarded/never-offloaded KV cache. `--max-num-
> batched-tokens` no longer needs to be held at 1024.

Investigated 2026-08-21/22, branch `request_logging`. This was a real bug
in upstream vLLM 0.27.1's `mamba_cache_mode=align` implementation, not a
config problem in this repo. A fix was written
(`patches/mamba-align-stale-state-queue.patch`) that correctly eliminates
the leak itself (verified, see "The fix" below), but every time it was
tried under fuller production conditions (real concurrency, offloading
enabled) it broke something else -- first an apparent hang (turned out to
be PSU-related, see below, not the fix), then a genuine KV-offloading
regression that turned out to be a second, separate, real upstream bug in
the connector (see "The fix breaks KV offloading at 2048" below) --
exposed by the leak fix (which made Mamba/GDN groups' offload paths
actually get exercised for the first time), but not caused by it. Both
are now fixed.

[← back to the main README](../README.md)

## Symptom

A single, unopposed ~88K-token prompt (`local_chat_test.py -p 4000`, no
second request, `--max-num-seqs 1`) would self-preempt one or more times
during its own first, uninterrupted prefill -- with nothing else
competing for GPU memory. GPU KV cache usage would climb to ~99-100% and
force a preemption well before raw token math predicted the pool should
fill up.

This was originally investigated as "does this only happen when two large
prompts race each other" (no -- reproduces solo), then "is it the KV
offload connector's bookkeeping" (no -- reproduces identically with the
connector removed entirely), before landing on the actual mechanism
below.

## Root cause

`--mamba-cache-mode align` (enabled by `PREFIX_CACHE=1` in
`start_qwen.sh`, which this setup wants for cross-turn prefix-cache
reuse) tracks each Mamba/GDN layer's recurrent state at block boundaries
instead of as a single O(1) blob, so a chunk that only partially advances
past a block boundary can still register a prefix-cache hit later. To
keep this from growing forever, `single_type_kv_cache_manager.py`'s
`MambaManager.remove_skipped_blocks()` is supposed to free the state
block from ~2 steps ago every step:

```python
# v1/core/single_type_kv_cache_manager.py, ~line 1436 (pre-patch)
last_state_block_idx = self.last_state_block_idx.get(request_id)
if (
    last_state_block_idx is not None
    and last_state_block_idx
    < cdiv(processed_computed_tokens, self.block_size) - 1
):
    blocks = self.req_to_blocks[request_id]
    if blocks[last_state_block_idx] != self._null_block:
        self.block_pool.free_blocks([blocks[last_state_block_idx]])
        blocks[last_state_block_idx] = self._null_block
```

`processed_computed_tokens` is `request.num_computed_tokens -
request.num_in_flight_tokens` (computed in `kv_cache_manager.py`'s
`allocate_slots()`), where `num_in_flight_tokens` is the async-scheduling
pipeline depth in tokens (tokens scheduled but whose output hasn't been
confirmed yet).

**During a continuous chunked-prefill run, `last_state_block_idx` and the
threshold `cdiv(processed_computed_tokens, block_size) - 1` increase in
lockstep and stay exactly equal on every single step**, so the strict
`<` comparison is never satisfied. Confirmed by direct instrumentation
(now removed -- it did its job; see the fix patch's changelog for the
proof, or re-add a similar log at the same call site to re-derive it):
logged every call to this check across a full 88K-token prefill, for all
3 of the model's Mamba/GDN KV-cache groups, at both
`max-num-batched-tokens=1024` and `=2048`:

```
will_free=True:  0
will_free=False: 107   (100% of all prefill-phase calls, both configs)
```

Sample line (note `last_state_block_idx == threshold`, never `<`):
```
last_state_block_idx=25 threshold=25 will_free=False
```

The check only starts passing once decode begins -- decode's per-step
chunk size (1-4 tokens, spec-decode verify) breaks the lockstep -- which
is after the entire prompt has already gone through prefill. So for the
whole duration of a large prompt's prefill, Mamba/GDN state blocks are
drawn fresh every step and **never freed**.

## Why chunk size determines whether this actually blows the budget

This model has **4 KV-cache groups sharing one block pool**, not the 1-2
you'd assume from "hybrid attention + mamba": 3 separate `MambaManager`
groups plus 1 `FullAttentionManager` group, all `block_size=832`.
Confirmed via `patches/preemption-per-group-alloc-debug-logging.patch`,
which logs each group's contribution to
`KVCacheCoordinator.get_num_blocks_to_allocate()`:

```
groups=[(0, 'MambaManager', 832, 1, N), (1, 'MambaManager', 832, 1, N),
        (2, 'MambaManager', 832, 1, N), (3, 'FullAttentionManager', 832, 2, N)]
total_new=5   free_blocks=<declining>
```

The *predicted* new-block count per step (`total_new`, the coordinator's
own estimate) is identical in shape at both configs -- 1 block/step per
Mamba group is a hardcoded cap in `MambaManager.allocate_new_blocks()`'s
align-mode branch (`num_new_blocks = 1 + int(has_partial_hit)` for an
already-running request), regardless of chunk size. What differs is
whether that predicted draw actually consumes a **real** pool slot.
Measured directly against `free_blocks` deltas, sampled over dozens of
consecutive steps each:

| `max-num-batched-tokens` | tokens/step | predicted new blocks/step | **actual** pool drain/step | real cost vs. attention-only |
|---|---|---|---|---|
| 1024 | 832 (1 block) | 4 (1 attn + 3×1 mamba) | **-1** | none -- matches pure attention, zero waste |
| 2048 | 1664 (2 blocks) | 5 (2 attn + 3×1 mamba) | **-5** | 2.5x -- all 3 mamba draws are real, unrecovered leaks |
| 8192 | 6656 (8 blocks) | (not fully measured) | -- | hard `torch.OutOfMemoryError`, engine crash |

At 1024, mamba's per-step "new block" request happens to get satisfied by
reclaiming an already-null-padded slot rather than drawing fresh from the
pool -- net cost is zero. At 2048, that reclaim doesn't happen, and 3
never-freed blocks get drawn per step on top of attention's own real 2.

**The alignment-dependent part traces to a *second*, separate freeing
mechanism** -- not the align-specific `last_state_block_idx` check above
(proven dead code during prefill), but the *base* class's
`SingleTypeKVCacheManager.remove_skipped_blocks()`, which Mamba's
`get_num_skipped_tokens()` override feeds almost the entire processed
length (`num_computed_tokens - 1`, since Mamba only needs its latest
state). That base method calls `_remove_blocks_in_range(request_id, 0,
num_skipped_blocks)`, which scans **backward** from
`num_skipped_blocks - 1` and **stops at the first null block it hits**.
This is the mechanism that was *actually* freeing mamba blocks all
along, purely by coincidence -- and whether it reaches the real
(non-null) stale block before hitting an intervening null from the
align-mode padding depends on how far `num_skipped_blocks` (itself
derived from the same lagged `processed_computed_tokens`) falls behind
the true block-list position, which is chunk-size sensitive: at 1024 it
reliably reaches (confirmed: 106/106 real frees, exactly balancing 106
real draws -- net zero growth); at 2048 it essentially never does
(confirmed: 3 real frees over 189 calls, vs. 5 real draws every step --
unbounded growth).

At 2048 (the previous default), this ~2.5x-worse-than-necessary rate
exhausted the pool after ~45 steps (~75K tokens) instead of comfortably
holding the full 88K-token prompt, and the request self-preempted
against *itself* -- no other request involved.

## Why a preempted request succeeds on retry

Not because anything gets fixed -- the retry just doesn't have enough
remaining distance to hit the wall again:

1. Preemption fully frees the leaked backlog (all draws made and never
   freed by the stalled check, across all 4 groups, get released).
2. The retry resumes from a large prefix-cache hit (observed: 69,888 of
   88,086 tokens, ~79%) -- that portion is reused via cache hit, not
   redrawn from the pool.
3. The retry only has to freshly process the remaining ~18K tokens
   (~11 steps at 2048). Even leaking at the same 5-blocks/step rate,
   11×5=55 blocks fits easily in the pool that was just cleared.

**A second, equally-long prompt with no cache hit would fail identically
against a clean pool.** This is a coincidence of remaining distance, not
a recovery of the underlying leak.

## The fix

`patches/mamba-align-stale-state-queue.patch`. Deliberately does **not**
touch the `<` comparison or `processed_computed_tokens` -- that threshold
is a genuine GPU-synchronization safety margin (`num_in_flight_tokens`
specifically excludes tokens whose GPU-side work isn't confirmed
complete), not an off-by-one, and loosening it risked trading this
preemption bug for a silent correctness bug (freeing Mamba state a
kernel might still be reading). Instead it fixes the *tracking* bug:
`last_state_block_idx: dict[str, int]` was a single scalar, overwritten
every step regardless of whether the *previous* stale position had
actually been freed. Since the freeing check stays in permanent lockstep
during continuous prefill (proven above), every step's overwrite
silently dropped the prior pending position before its own check ever
got a chance to pass.

Turned into a per-request FIFO queue (`_stale_state_block_idxs: dict[str,
list[int]]`) instead. Same threshold, same safety margin, applied to
every pending entry rather than only the newest. Because indices are
strictly increasing and the newest one sits exactly *at* the threshold
every step (the lockstep, proven above), every *older* pending entry is
therefore already strictly *past* the threshold the moment the newest
one is checked -- so it gets freed immediately. The queue stays bounded
at ~1 pending entry instead of growing with the prompt, and the
coincidental base-class fallback mechanism (the `_remove_blocks_in_range`
scan described above) goes back to being irrelevant to Mamba, exactly as
it was presumably intended to be.

**Verified** (2026-08-21/22, `MAX_SEQS=1`, no other request involved):
the same 88K-token solo prompt that self-preempted at
`--max-num-batched-tokens=2048` (peak GPU-KV usage 99.6%, self-preempted)
now peaks at **54.9%** with **zero preemptions**, at the same batch size
-- essentially matching the pure-attention-cost expectation
(ceil(88086/832) = 106 blocks needed; 124 observed, ~17% residual
overhead, same small margin the 1024 mitigation achieved). Throughput
improved too (885 vs. ~860 tok/s), since this also removes the reason
`--max-num-batched-tokens` had to be held down to 1024 in the first
place. Correctness spot-checked with a needle-in-haystack recall test
(a unique 6-digit code embedded ~44K tokens into an 88K-token prompt,
greedy decoding) -- correctly recalled.

**`--max-num-batched-tokens` no longer needs to be held at 1024** for
this reason. Whether `single-user/start_qwen.sh` has actually been
reverted to 2048 is a separate question from whether the fix works --
check the file's current value rather than assuming.

**Do not** try `--max-num-batched-tokens 8192` or higher without
re-checking `--gpu-memory-utilization` headroom first, independent of
this fix -- it caused a hard `torch.OutOfMemoryError` engine crash in
testing (profiled activation memory at that batch size left ~70MB less
room than the KV cache needed at the time; bumping `GPU_UTIL` to 0.94
worked around the startup check, but real per-step memory pressure at
8192 still crashed the engine mid-request -- unrelated to the mamba leak,
this is pure activation-memory pressure from the larger chunk).

## The suspected hang -- re-tested clean after a hardware fix (this part held up)

**Note:** this section's conclusion (hang was PSU, not the fix) has held
up fine -- it's the *next* section ("The fix breaks KV offloading") that
found the actual reason the fix can't be applied yet. Keeping this
section as-is for the record.

**Update 2026-08-22, ~10:15 UTC:** after the machine was rebooted for an
unrelated GPU fault (Xid 154, "Node Reboot Required" -- hit mid-retest,
see the incident note below), the user identified the actual cause as a
**failing PSU**, since fixed. With that hardware issue addressed, the
exact hang repro from the section below was re-run from a clean restart
(fresh `micke-start.sh`, `--max-num-seqs 8` + `OffloadingConnector`, fix
patch applied) and did **not** reproduce:

- Multi-turn `local_chat_test.py -p 4000` repro, run twice: both times
  completed cleanly end to end (turn 1 ttfb~100s as expected for an 88K
  prompt, turn 2 ttfb~3-5s off a strong prefix-cache hit -- no
  reprocessing, no freeze). `qwen.log` showed `Running: 0 reqs` shortly
  after each turn's generation ended, not a stuck heartbeat.
- Two concurrent solo 88K-token prompts (the scenario that used to
  self-preempt against the pool, matching what was captured in
  `qwen.log` right at reboot, under the *original leaky* code): both
  completed with **zero preemptions each** (`num_preemptions=0` in the
  `_free_request_blocks` trace for both), peak GPU-KV usage **55.6%**
  even with two 88K prompts in flight simultaneously -- a single 88K
  prompt alone used to peak the pool at 99.6% under the old leaky code.

No hang, no stuck heartbeat lines, no CPU-pinned `VLLM::EngineCore`
across 4 separate runs. This is the same repro that froze solid before
the reboot (frozen at 11.1% GPU-KV usage, `VLLM::EngineCore` at 100% CPU
in state `R`, per the original writeup preserved below).

**This does not conclusively rule out the original hypothesis** (a race
between the per-step Mamba freeing path and the offloading connector's
in-flight job tracking, described below) -- the hang was never actually
caught with a stack trace, so there's no proof it was the PSU rather
than the fix. But given: (a) a dying PSU under sustained GPU load is a
plausible, independently-confirmed-real cause of exactly this kind of
symptom (a wedged process that looks like a busy-loop rather than a
clean crash), (b) the user identified and fixed a real PSU problem
around the same time, and (c) 4/4 clean re-runs of the exact repro
afterward, the balance of evidence now favors "hardware, not the fix."
**Action taken:** re-applied `patches/mamba-align-stale-state-queue.patch`
to the live venv (confirmed via `grep -c _stale_state_block_idxs`
returning non-zero). The fix is live as of this update.

**If a hang like this happens again** (frozen heartbeat, `GPU KV cache
usage` stuck, `VLLM::EngineCore` pinned at 100% CPU in state `R`, no new
`Running batch` lines): that would be much stronger evidence the fix
really does have the race described below, now that a hardware
explanation has been tested and didn't reproduce it. Two things are now
in place to make that diagnosis faster next time, in case it does:
- `venv/lib/python3.12/site-packages/sitecustomize.py` was added (not a
  vLLM patch, a venv-local file, so it won't show up in `git diff`
  against the patches) -- it registers a `SIGUSR1` handler via
  `faulthandler` on every Python process started from this venv,
  including `VLLM::EngineCore`. If it hangs again: find the EngineCore
  PID (`ps aux | grep VLLM::EngineCore`), `kill -USR1 <pid>`, then read
  `/tmp/vllm_faulthandler_dump.txt` for a full all-threads Python
  traceback -- no `sudo`/`py-spy`/`ptrace_scope` needed, this works as
  the same unprivileged user. This is the single highest-value tool for
  actually confirming or killing the race hypothesis below.
- `py-spy` was pip-installed into the venv in the prior session but
  couldn't attach without root; the `sitecustomize.py` hook above is a
  better fit for this environment (no `ptrace_scope`/sudo dependency) and
  should be tried first.

The original hang investigation from the pre-reboot session is preserved
below verbatim for context (the working hypothesis about the offloading
connector race is unconfirmed either way -- neither proven nor
disproven by today's clean re-runs, since a hardware fault could easily
have been masking or mimicking it).

### Original hang writeup (pre-reboot, PSU issue not yet identified)

Found 2026-08-22, ~07:30 UTC, by the user testing the fix under normal
production config (`micke-start.sh` → `--max-num-seqs 8`,
`--kv-transfer-config` with `OffloadingConnector`/`TieringOffloadingSpec`
-- **not** the `MAX_SEQS=1`, later no-connector, isolated configs the fix
was verified under above). Symptom: a multi-turn `local_chat_test.py -p
4000` session froze completely partway through the first turn's prefill
and never recovered ("it probably won't complete on its own if it's stuck
in a loop" -- correct).

**Confirmed facts, in order of investigation:**

1. `qwen.log` showed the request frozen at exactly 11.1% GPU-KV usage,
   heartbeat lines (`loggers.py:468`) ticking once per second forever
   with `Avg prompt throughput: 0.0 tokens/s`, no new `gpu_model_runner.py
   Running batch` line after the freeze point -- the forward-pass loop
   had stopped issuing steps entirely.
2. The freeze started immediately after `Request X offloading 5 chunks
   upto 6656 tokens (job 3)` -- i.e. right as the 4th offload job for
   this request was queued, very early in prefill.
3. `ps`/`top` showed the `VLLM::EngineCore` process at **100% CPU,
   state R (actively running, not blocked/sleeping)** -- this is a
   genuine busy-loop, not a deadlock waiting on I/O or a lock, and not a
   crash.
4. `patches/discard-unoffloaded-kv-warning.patch`'s new code
   (`_warn_if_discarding_unoffloaded_kv`) only runs from
   `_free_request_blocks()`, which only fires on request finish/abort/
   preemption. The last such call in `qwen.log` was ~6.5 minutes *before*
   the freeze started, and none occurred during it -- **ruled out** as
   the cause.
5. That leaves `patches/mamba-align-stale-state-queue.patch`'s
   `remove_skipped_blocks()` change, which runs on *every scheduling
   step* (not just at finish time) -- the prime suspect by elimination.
6. Checked `distributed/kv_transfer/kv_connector/v1/offloading/
   scheduler.py` directly: Mamba/GDN KV-cache groups **are** included in
   the offloading connector's tracked groups
   (`resolve_mamba_align_size()` scans all groups including Mamba ones),
   and the connector's own store-job logic explicitly handles "null
   placeholder blocks used for sliding window or mamba padding" -- so
   Mamba blocks are genuinely part of in-flight offload jobs, not just
   attention blocks.

**Working hypothesis, not yet confirmed by a stack trace:** the queue fix
makes Mamba state blocks get freed far earlier and far more often than
the original (effectively-never-frees-during-prefill) code ever did.
Unlike the *scheduler*-level preemption fix
(`patches/preemption-defer-block-free.patch`, `_free_request_blocks(...,
force_defer=...)`), which explicitly defers freeing a preempted request's
blocks until the connector's `jobs_to_flush` for that request have
drained, this per-step Mamba freeing path lives deep inside
`SingleTypeKVCacheManager` / `KVCacheManager.allocate_slots()` and has
**no equivalent guard** against freeing a block that's still part of an
in-flight offload job. The connector does have its own protection for
this in general (`_block_id_to_pending_jobs`, triggering a flush when a
pending-job block is about to be *reused*) -- but that's triggered from
the allocation side, and it's plausible the interaction with freeing from
this specific, newly-exercised path leaves some tracking structure
inconsistent, causing a wait/retry loop that spins forever instead of
resolving. **This was not confirmed with a live stack trace** -- `py-spy`
had to be installed fresh and then couldn't attach without root
(`ptrace_scope=1`, process wasn't a direct child, no sudo password
available in the session). Getting an actual stack trace of the spinning
`VLLM::EngineCore` process (`sudo py-spy dump --pid <pid>`, or `sudo
gdb -p <pid>` with the python extension, `py-bt`) while it's hung is the
single highest-value next step -- it would turn this from a hypothesis
into a confirmed root cause.

**Action taken:** reverse-applied `patches/mamba-align-stale-state-queue.
patch` against the live venv (`single_type_kv_cache_manager.py`) --
confirmed via `grep -c _stale_state_block_idxs` returning 0 after the
revert. Original scalar `last_state_block_idx` code (leaky but hang-free)
is what's live. `patches/discard-unoffloaded-kv-warning.patch` was left
applied (ruled out above, and independently useful). **The revert was not
re-verified to fix the hang before the machine had to be rebooted** for
an unrelated reason (GPU stuck at ~23.6GB used after force-killing the
hung process -- `fuser`/process list showed nothing holding it, driver-
level cleanup issue from the cumem allocator not releasing on SIGKILL,
needed a reboot to clear).

**Next steps in order (as planned pre-reboot -- steps 1-2 are now DONE,
see the update above; kept here for the historical record):**
1. ~~After reboot, confirm GPU is clear and confirm the revert is in
   place.~~ Done.
2. ~~Re-run the exact repro with the fix reverted, to confirm the hang is
   actually gone with the original code.~~ Superseded: the PSU was
   identified as the likely real cause before this step happened, so
   instead the fix was re-applied and *the fix itself* was re-tested
   (see update above) -- it came back clean 4/4, so this step never
   ended up being needed to isolate the variable.
3. Still open, only relevant **if a hang recurs**: get a real stack trace
   while it's hung. Now doable without root or `py-spy` -- see the
   `sitecustomize.py` / `SIGUSR1` / `faulthandler` mechanism described in
   the update above (`kill -USR1 <EngineCore pid>`, read
   `/tmp/vllm_faulthandler_dump.txt`).
4. Still open, only relevant if step 3 ever confirms a real race: the
   corrected fix would likely need the Mamba per-step freeing path to
   either (a) consult `_block_id_to_pending_jobs` (or equivalent) before
   calling `block_pool.free_blocks()`, deferring like the preemption fix
   does, or (b) not call `free_blocks()` directly from this layer at all
   and instead route through whatever mechanism the scheduler-level fix
   uses.

## The fix breaks KV offloading at 2048 -- root cause found and fixed

Found 2026-08-22, ~11:00 UTC, extensively re-investigated same day through
~13:40 UTC. After the hang was provisionally cleared (previous section),
the fix was re-applied and left running in production. The user then
reported: *"RAM/Disk offloading doesn't seem to work anymore... tons of
reprocessing when everything should be in cache."* This is a **different
bug from the hang** -- no freeze, no stuck heartbeat, the engine makes
normal progress throughout -- but a multi-turn conversation that should
get a fast RAM/NVMe cache hit instead does a full, ~100-second-per-88K-
tokens reprocess from scratch. This directly explains why the *original*
pre-reboot report ("we are now _re-processing_ prompts _after_ the reply
is finished") looked like an infinite loop: a full reprocess of an 88K
prompt just looks like a stall if you're not watching the token counter.

**Bottom line up front:** this was real, 100% reproducible, and is now
**fixed** (`patches/offload-eagle-misclassification-mamba.patch`) -- a
genuine, separate bug in vLLM's own offloading connector, not in the leak
fix. It was misleadingly tied to `--max-num-batched-tokens` (broken at
2048, working at 1024) during the investigation -- that correlation was
real but was a red herring for the actual mechanism (see "What it is:
tied to chunk size" below for how that led toward, but not quite to, the
real cause; the actual root cause has nothing to do with chunk size or
timing at all, see "Root cause found" further down). A large amount of
work went into ruling out simpler explanations one at a time first; each
is recorded below because re-deriving them cost real time and they are
all non-obvious, even though the final answer turned out to be none of
them.

### Fast, reliable repro

`test_offload_regression.sh` (repo root). Runs 2 sequential 88K-token
conversations against a running server -- conv A turn 1, then conv B
turn 1 (big enough that B evicts A's GPU-local prefix cache), then conv A
turn 2, which should be served from the RAM/NVMe offload tier if
offloading is working:

```
bash micke-start.sh                    # or whatever config you're testing
# wait for /health, then:
bash test_offload_regression.sh
```

**Working** (fix reverted, OR fix applied at `--max-num-batched-tokens
1024`): conv A turn 2 finishes in **4.6-8s**, and `qwen.log` shows the
connector's own hit-confirmation line, e.g. `Request chatcmpl-... hit
85696 offloaded tokens after 0 GPU hit tokens`, confirmed by GPU-KV usage
having hit 100% in between (so it wasn't just a lucky local-cache
survival, it's a genuine external hit).

**Broken** (fix applied at the default `--max-num-batched-tokens 2048`):
conv A turn 2 takes **~100-105s** (identical throughput to a cold
first-turn prefill), and `qwen.log` has **zero** `"offloaded tokens
after"` lines for that request. Total miss, every time. Reproduced well
over a dozen times across this investigation -- 100% reliable at 2048,
100% working at 1024, not intermittent either way.

### What it is not (each ruled out with a dedicated, isolated test)

- **Not disk exhaustion.** `/d/nvme_cache` genuinely did fill to 100%
  during this investigation (the fs secondary tier has no size cap or
  eviction at all -- see the new TODO item at the bottom of this doc) and
  caused real `[Errno 28] No space left on device` errors for a while,
  which looked like a plausible cause. But the failure reproduces
  identically with the disk freshly cleared (16K used / 222G free) --
  disk pressure was a real, separate bug (worth fixing on its own) that
  happened to overlap in time, not the cause of this one.
- **Not a free-before-store race.** Instrumented the connector's
  `_build_store_jobs()` at its `if block_id == 0: continue` skip point
  (the only place a freed/nulled Mamba block could make a chunk
  unstorable) directly: **zero skips**, for either Mamba or attention
  groups, across every test run. The physical bytes are always
  successfully captured before anything reuses that GPU memory.
- **Not free-vs-store timing/margin, at any margin.** Tried holding back
  the newest 1, and then the newest 20, pending stale-block entries
  before ever calling `block_pool.free_blocks()` on them (i.e.
  progressively more conservative versions of "wait before freeing").
  **Zero effect** on the outcome at either margin -- still a total miss
  every time at 2048. This rules out any theory shaped like "the fix
  frees a block N steps too early."
- **Not block-hash assignment timing.** Hypothesis: `cache_blocks()`
  (which assigns `block.block_hash`, and runs from `AsyncScheduler.
  _update_request_with_output()` -- triggered when a step's GPU results
  return, a separate and *later* event than `processed_computed_tokens`
  crossing a block's boundary) might not have run yet for a block by the
  time our fix frees it, corrupting the hash chain the connector's
  offload keys derive from. Gated freeing on `block.block_hash is not
  None` directly (the precise, unambiguous version of this check, not a
  proxy). **Zero effect** -- still fails identically. So either the hash
  is already assigned by the time our check runs, or this isn't the
  mechanism.
- **Not the switch from a scalar to a list, independent of freeing
  behavior.** Kept the new `_stale_state_block_idxs: dict[str, list[int]]`
  tracking structure, but made the actual freeing behave byte-for-byte
  like the original scalar (only ever consider the single newest entry,
  clear everything else, matching the original's "never actually frees
  during continuous prefill" behavior). This **worked** (fast hit) at
  2048 -- but this only proves "freeing nothing beyond what the original
  code froze" is safe, which is a tautology (it's not exercising the fix
  at all). See the next point for why this doesn't mean margin-based
  throttling is the fix either.

### What it is: tied to chunk size, not to any freeing condition

The decisive experiment: run the **actual, unthrottled fix** (frees a
block the instant it's safely stale, no artificial margin) at
`--max-num-batched-tokens 1024` instead of 2048. Result: **works**
(`hit 86528 offloaded tokens`, 6s turn-2). Same fix code, only the batch
size changed. So:

| | 1024 | 2048 |
|---|---|---|
| Leak fixed (peak GPU-KV usage) | yes | yes |
| Offloading works | **yes** | **no** |

This rules out every freeing-condition theory above at once (they'd have
to also explain why the *identical* freeing logic works fine at 1024) and
points somewhere else: **the interaction is with request throughput /
scheduling-step frequency, not with any per-block timing decision our
code makes.**

Direct evidence for the mechanism, from tagging the connector's own
per-group lookup result (`_lookup()`'s `num_hit_chunks`, scheduler.py
~702-729) with `group_idx` and comparing the *same* request side by side:

```
# WORKING (fix reverted, 2048):
group_idx=3 (attention)  -> 105/106 hit
group_idx=0 (mamba)      -> 104/105 hit   <- one chunk short, same as always
group_idx=1 (mamba)      -> 104/104 hit
group_idx=2 (mamba)      -> 104/104 hit

# BROKEN (fix applied, 2048):
group_idx=3 (attention)  -> 105/106 hit   <- identical to the working case
group_idx=0 (mamba)      -> 0/105 hit     <- clean, total miss
```

Full-attention's own lookup is **identical** in both cases (105/106,
tolerating one unresolved/pending trailing chunk fine -- its
`_maximal_prefix_lookup` just returns however much of a run it confirmed).
Mamba's lookup goes through `_sliding_window_lookup()` instead (window
size 1, since Mamba groups get `sliding_window_size_in_chunks=1`
regardless of `mamba_cache_mode` -- see `get_sliding_window_size_in_chunks()`,
scheduler.py:107). That function scans backward and does **not** break on
a `RETRY` result (deliberately, "to let manager kick off async lookups") --
but critically, if *any* entry newer than a later confirmed hit came back
`RETRY`, the whole scan's result is discarded to `None` (deferred)
regardless of the hit (`return idx + sliding_window_size if not
defer_lookup else None`, scheduler.py:609). So Mamba's lookup has **zero
tolerance** for an unsettled recent chunk, where attention's has
essentially unlimited tolerance (one confirmed run is enough, however far
back).

Mamba's own align-mode block churn is **step-count-driven, not
token-count-driven**: `allocate_new_blocks()`'s align-mode branch grants
`num_new_blocks = 1` per group per scheduling step, hardcoded, regardless
of `--max-num-batched-tokens` (see "Why chunk size determines..." above).
At 2048 vs. 1024, roughly the same wall-clock throughput (~800-900 tok/s)
is achieved with **half as many, twice-as-big scheduling steps** -- so
Mamba's per-*step* churn rate is unchanged, but the number of
opportunities-per-second for whatever polls/settles the offloading
connector's async promotion pipeline (`TieringOffloadingManager.
_maybe_process_finished_jobs()`, gated to run at most once per step) is
roughly **halved**. The working theory, not yet proven down to the exact
stall: at 2048 the async promotion pipeline doesn't get polled often
enough (in real time) to keep pace with Mamba's step-driven churn, so its
newest 1-2 chunks are still showing `RETRY` (promotion in flight, not yet
confirmed) at the exact moment a fresh request's lookup scans them --
and because of the zero-tolerance behavior above, that alone sinks the
*entire* Mamba group's result, discarding an otherwise-fine hit found
further back. Full-attention's own chunks presumably hit the same
`RETRY` states sometimes too, but its lookup tolerates them fine.

This was directly observed once, per-key, before the group-level
comparison above made it unnecessary to keep digging further: for one
`RETRY`-heavy scan, indices 105 and 104 (the two newest) returned `RETRY`,
then index 103 returned a clean `HIT` -- and the function's own semantics
discarded the result to `None` anyway because of the two RETRYs ahead of
it (see the code excerpt above). On a *later* rescan (after the query
window had narrowed following the attention group's own successful
lookup), index 104 had settled to a clean `MISS` and index 103 was still
`HIT` -- but by then the query window had already been narrowed past 105
by the outer loop, and the group as a whole still failed for other
reasons visible only in the full trace. The exact stall was not chased
further than this once the batch-size experiment above made the
higher-level mechanism (step-frequency-driven polling vs. token-count-driven
churn) clear enough to stop guessing at the last mile.

### What was NOT the cause (revised from earlier, wrong hypotheses)

Two earlier hypotheses in this section, kept here so they aren't
re-derived and re-disproven again:

- A same-scheduling-step race between our fix's `block_pool.free_blocks()`
  call (in `allocate_slots()`, which runs before `build_connector_meta()`
  within `Scheduler.schedule()`) and the connector's store-job builder
  seeing a null block. Directly instrumented and **disproven**: zero
  `block_id == 0` skips ever observed in `_build_store_jobs()`, at any
  margin, in any run.
- The step-frequency/async-promotion-polling theory in the section just
  above ("What it is: tied to chunk size"). This correctly identified
  that batch size correlated with the bug and correctly ruled out every
  freeing-condition theory, but the *mechanism* it proposed (promotions
  not settling in time at low step-frequency) was never actually
  confirmed down to a stack trace or a stalled job -- and turned out not
  to be it. See below for what was actually happening, which also
  explains *why* 1024 happened to avoid it (smaller chunks happen to keep
  a stale intermediate query-window state from ever narrowing into the
  failure condition as reliably -- not because anything was settling
  faster).

### Root cause found: Mamba/GDN groups misclassified as EAGLE/MTP draft groups

Found by instrumenting `_lookup()`'s per-group result with `group_idx`,
`num_hit_chunks`, and (crucially, the piece the earlier investigation
hadn't checked) `is_eagle_group`/`is_eagle_unverified`:

```
group_idx=3 (attention) num_hit_chunks=105 is_eagle_group=True is_eagle_unverified=True
group_idx=0 (mamba)     num_hit_chunks=0   is_eagle_group=True is_eagle_unverified=True
```

**`is_eagle_group=True` for the Mamba/GDN groups was the bug.**
`SchedulerOffloadConfig.from_spec()` (scheduler.py, ~line 208) has this
fallback:

```python
eagle_groups = {idx for idx, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
use_eagle = vllm_config.speculative_config is not None and vllm_config.speculative_config.use_eagle()
if use_eagle and not eagle_groups:
    eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))  # <- marks EVERY group
```

This model's KV-cache groups aren't explicitly tagged `is_eagle_group`
(correctly -- none of them, including the 3 Mamba/GDN groups, are the MTP
draft head's own state; this repo's MTP draft module has no mamba
layers). Since `speculative_config.use_eagle()` is True for MTP and
`eagle_groups` came back empty, the fallback assumed *every* group might
hold volatile draft-model state and marked all 4 groups -- including the
3 Mamba/GDN ones -- as eagle groups.

Being an eagle group adds +1 to the required window in
`_sliding_window_lookup()` (`required_window = sliding_window_size_in_chunks
+ 1` when `is_eagle_unverified`, to query one extra provisional chunk and
pop it once verified -- a real mechanism, needed for genuine draft-model
groups). For Mamba, this turned a "need 1 hit" lookup into a "need 2
*consecutive* hits" lookup. Per-key instrumentation showed the actual
data alternates hit/miss/hit/miss with no two consecutive hits ever
(itself a downstream effect of the same eagle-only "extra provisional
chunk" querying, applied somewhere it structurally doesn't have a partner
chunk to pair with) -- so this lookup was **guaranteed to fail every
single time**, at any batch size, independent of any timing or race
condition. Full-attention's group is also marked (wrongly) as eagle, but
`_maximal_prefix_lookup()` (used for non-sliding-window groups) doesn't
have a consecutive-run requirement, so the misclassification is silent
there -- which is exactly why only Mamba's lookup ever showed a problem.

Batch size was a real but incidental correlate: it changed how the
`_lookup()` convergence loop's query-window narrowing happened to line up
against the alternating hit/miss pattern across different runs, which is
why 1024 looked reliably safe in testing -- not because it fixed anything
about the actual defect.

**The fix** (`patches/offload-eagle-misclassification-mamba.patch`):
exclude `MambaSpec` groups from the eagle-fallback classification --
they're never a legitimate "draft model's volatile trailing chunk"
scenario the way a shared/ambiguous attention KV group can be. Includes a
second, smaller, defense-in-depth fix in the same function:
`_sliding_window_lookup()` previously discarded an already-confirmed HIT
run if *any* newer position scanned first came back `RETRY`, even though
that older run was already safe to use (the mirror image of the eagle
bug -- an older valid hit shouldn't be held hostage by a newer, unrelated
position's settling status). This didn't turn out to be the primary
cause here (the eagle miscount alone was sufficient to explain 100% of
failures), but it's a real, independently-reasoned correctness
improvement, kept in the same patch. See the patch file's own header for
the full writeup.

**Verified** (2026-08-22, both patches applied, default
`--max-num-batched-tokens 2048`): `test_offload_regression.sh` passes
reliably across multiple runs (`hit 84864 offloaded tokens`-style lines,
turn 2 in 6-8s). 2x genuinely concurrent ~88K-token prompts complete with
zero preemptions. A 3x concurrent ~120K-token stress test (360K tokens of
combined demand against a ~172K-token pool, over 2x oversubscribed)
completes with **zero preemptions and zero discarded/never-offloaded KV
cache**, peak GPU-KV usage only 72.4%. The leak fix's own numbers are
unaffected (peak usage ~55%, zero preemptions on the original solo-88K-
prompt repro).

### Action taken

Both patches applied and verified:
`patches/mamba-align-stale-state-queue.patch` (the leak fix) and
`patches/offload-eagle-misclassification-mamba.patch` (the offloading
fix). `single-user/start_qwen.sh` stays at `--max-num-batched-tokens
2048` (it was temporarily flipped to 1024 for a diagnostic experiment
during the investigation and restored via `git checkout`). All debug
instrumentation added during this investigation (temporary `print()`
calls across `single_type_kv_cache_manager.py`, `distributed/kv_transfer/
kv_connector/v1/offloading/scheduler.py`, and `v1/kv_offload/tiering/
manager.py`) was removed before finalizing the patches -- both patch
files are minimal, confirmed via `patch -p1 -R --dry-run` matching the
live tree exactly.

### Separate TODO surfaced by this investigation: no disk cap on the fs offload tier

`v1/kv_offload/tiering/fs/manager.py`'s `FileSystemTierManager.__init__`
takes `root_dir`, `n_read_threads`, `n_write_threads`,
`enable_kv_events`, `locality` -- **no size limit, no eviction policy**,
unlike the primary CPU tier (`cpu_bytes_to_use`, LRU/ARC eviction). It
just writes files to `root_dir` forever. This investigation's repeated
88K-token test runs filled `/d/nvme_cache` (234G) to 100% over the
course of a few hours, which is exactly what would eventually happen in
normal long-running production use too, just slower. No config flag
exists to cap it today (confirmed by reading the full `__init__`
signature and its `SecondaryTierFactory` construction path -- any extra
key in `kv_connector_extra_config.secondary_tiers[...]` besides `type`
gets passed straight through as a kwarg, and there's nothing to catch).
Worth a follow-up: either an upstream vLLM feature (real size-based LRU
eviction for the fs tier, mirroring the primary tier's), or in the
meantime an external prune script/cron/systemd-timer against `root_dir`
as a practical mitigation.

## Reproducing the hang (historical -- only relevant if a hang recurs; both fixes are applied as of 2026-08-22 ~15:30 UTC, see the offloading section above)

```
bash micke-start.sh
```
(normal production config: `--max-num-seqs 8`, `--kv-transfer-config`
with `OffloadingConnector`/`TieringOffloadingSpec`, `CTX=long`,
`MAX_LEN=140000` -- this is the config that *appeared* to hang once,
pre-PSU-fix; the isolated `MAX_SEQS=1` / no-connector configs used to
verify the fix originally did **not** reproduce it either way)

Then, against a fresh server, a genuine multi-turn session (not a single
solo probe):
```
printf "Please summarize the key themes in the reference material above in about 3 sentences.\nThanks - now, in one sentence, what did you just say?\n" \
  | venv/bin/python local_chat_test.py -p 4000
```

Watch `qwen.log` for `Avg prompt throughput: 0.0 tokens/s` heartbeat
lines repeating with a frozen `GPU KV cache usage` and no new
`gpu_model_runner.py Running batch` lines -- that's the freeze. Confirm
with `top`/`ps` that the `VLLM::EngineCore` process is pinned at 100% CPU
in state `R` (busy-loop, not blocked). Re-run 2026-08-22 (~10:15 UTC)
with `patches/mamba-align-stale-state-queue.patch` applied and it
completed cleanly twice in a row, plus a two-concurrent-88K-prompt
variant with zero preemptions -- see "The suspected hang" section above
for exact numbers. The patch is reverted again now (see the offloading
section above), for an unrelated reason -- the hang itself has not
recurred. If `kill -USR1 <EngineCore pid>` + reading
`/tmp/vllm_faulthandler_dump.txt` is ever needed, the handler is
registered via `venv/lib/python3.12/site-packages/sitecustomize.py`.

## Reproducing the original memory leak

```
MAX_SEQS=1 ITERATION_LOG=1 VLLM_LOG_STATS_INTERVAL=1 PREFIX_CACHE=1 \
CTX=long MAX_LEN=140000 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-cumem-allocator --max-num-batched-tokens 2048' \
bash single-user/start_qwen.sh
```
(omit `--kv-transfer-config` / offloading to isolate from that connector
entirely -- confirmed the leak itself is unrelated to it, independent of
the hang)

Then, against a fresh server:
```
echo "..." | venv/bin/python local_chat_test.py -p 4000
```

This is what `patches/mamba-align-stale-state-queue.patch` fixes. Both
that patch and `patches/offload-eagle-misclassification-mamba.patch` are
applied in the current venv -- reverse-apply
`mamba-align-stale-state-queue.patch` first if you want to see the
original leak/self-preemption behavior again.

Diagnostic patches (apply on top of the base patch set, quiet by default
-- gated on free blocks < 15% of pool, so normal short-prompt traffic
won't spam `qwen.log`):
- `patches/preemption-per-group-alloc-debug-logging.patch` -- per-KV-group block allocation breakdown
- `patches/preemption-usage-debug-logging.patch` -- raw block-pool usage numbers
- `patches/preemption-lookup-debug-logging.patch` -- prefix-cache hit sizes on preempted-request retries

The freeing-check and `_remove_blocks_in_range` traces that proved the
lockstep and the base-class fallback mechanism (respectively) were
removed once they'd done their job; re-add a similar `logger.debug()` at
`remove_skipped_blocks()` / `_remove_blocks_in_range()` in
`single_type_kv_cache_manager.py` if re-deriving that proof is ever
needed again.

grep `qwen.log` for `request-logging preemption-debug:` to find the
diagnostic patches' output.
