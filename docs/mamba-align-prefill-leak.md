# Mamba-align prefill memory leak (self-preemption on large single prompts)

> **STATUS as of 2026-08-22, ~07:35 UTC: fix REVERTED, DO NOT re-apply
> `patches/mamba-align-stale-state-queue.patch` without reading "The fix
> caused a production hang" section below first.** The fix correctly
> eliminates the leak (verified) but caused a real hang under production
> config (`--max-num-seqs 8` + `--kv-transfer-config` offloading) that
> wasn't exercised by the verification testing at the time. The live venv
> currently has the patch reverted -- original leaky-but-safe code is
> running. Machine was rebooted after this was found (unrelated GPU
> driver issue from force-killing the hung process, not a fix side
> effect) -- see that section for exact repro steps and where the
> investigation was cut off. Read that section fully before touching this
> patch again.

Investigated 2026-08-21/22, branch `request_logging`. This is a real bug
in upstream vLLM 0.27.1's `mamba_cache_mode=align` implementation, not a
config problem in this repo. A fix was written
(`patches/mamba-align-stale-state-queue.patch`) and initially verified,
then found to cause a hang under fuller production conditions -- see
below. Until a corrected fix lands, the safe mitigation remains holding
`--max-num-batched-tokens` at 1024 (see "Current mitigation" below --
note this section predates the fix attempt, check `start_qwen.sh`'s
actual current value rather than assuming).

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

## The fix caused a production hang -- REVERTED, not currently applied

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

**Next steps in order:**
1. After reboot, confirm GPU is clear (`nvidia-smi`) and confirm the
   revert is still in place in the venv (`grep -c _stale_state_block_idxs
   venv/lib/python3.12/site-packages/vllm/v1/core/single_type_kv_cache_manager.py`
   should print `0`) -- reboot shouldn't have touched it (it's a live
   edit to installed package files, not a running process), but verify.
2. Re-run the exact repro (multi-turn `local_chat_test.py -p 4000`
   against a `micke-start.sh`-launched server, i.e.
   `--max-num-seqs 8` + offloading) with the fix reverted, to confirm the
   hang is actually gone with the original code -- this hasn't been
   directly confirmed yet, only inferred from reverting the prime
   suspect.
3. If confirmed gone: get a real stack trace of the *fixed* code hanging
   (re-apply the patch, reproduce again, `sudo py-spy dump`) to nail the
   exact mechanism before attempting a corrected fix.
4. The corrected fix likely needs the Mamba per-step freeing path to
   either (a) consult `_block_id_to_pending_jobs` (or equivalent) before
   calling `block_pool.free_blocks()`, deferring like the preemption fix
   does, or (b) not call `free_blocks()` directly from this layer at all
   and instead route through whatever mechanism the scheduler-level fix
   uses. Needs someone to trace how `_block_id_to_pending_jobs` gets
   populated/consumed and whether `SingleTypeKVCacheManager` even has a
   handle on the connector to check it from -- it may not, structurally,
   which would mean the real fix has to move up a layer.

## Reproducing the hang (current priority)

```
bash micke-start.sh
```
(normal production config: `--max-num-seqs 8`, `--kv-transfer-config`
with `OffloadingConnector`/`TieringOffloadingSpec`, `CTX=long`,
`MAX_LEN=140000` -- this is the config that hung; the isolated
`MAX_SEQS=1` / no-connector configs used to verify the fix originally did
**not** reproduce it)

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
in state `R` (busy-loop, not blocked). This requires
`patches/mamba-align-stale-state-queue.patch` to be **applied** (it's
reverted by default right now -- see status banner at the top).

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

This is what the fix (when it worked) resolved. The patch is currently
reverted, so this config will currently show the original leak/
self-preemption behavior, not the hang -- the hang needs the offloading
connector + real concurrency, per the section above.

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
