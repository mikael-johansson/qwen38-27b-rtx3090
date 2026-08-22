# Mamba-align prefill memory leak (self-preemption on large single prompts)

Investigated and **fixed** 2026-08-21/22, branch `request_logging`. This
was a real bug in upstream vLLM 0.27.1's `mamba_cache_mode=align`
implementation, not a config problem in this repo. Fixed in
`patches/mamba-align-stale-state-queue.patch`; `--max-num-batched-tokens`
no longer needs to be held down to 1024 to avoid it (that was a
mitigation, applied and then superseded once the real fix landed -- see
below for whether `start_qwen.sh` has been reverted to 2048).

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

## Reproducing

```
MAX_SEQS=1 ITERATION_LOG=1 VLLM_LOG_STATS_INTERVAL=1 PREFIX_CACHE=1 \
CTX=long MAX_LEN=140000 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
EXTRA_ARGS='--enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-cumem-allocator --max-num-batched-tokens 2048' \
bash single-user/start_qwen.sh
```
(omit `--kv-transfer-config` / offloading to isolate from that connector
entirely -- confirmed the leak is unrelated to it)

Then, against a fresh server:
```
echo "..." | venv/bin/python local_chat_test.py -p 4000
```

To reproduce the *original* (pre-fix) bug rather than confirm the fix,
revert `patches/mamba-align-stale-state-queue.patch` before starting the
server.

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
