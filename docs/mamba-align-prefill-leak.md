# Mamba-align prefill memory leak (self-preemption on large single prompts)

Investigated 2026-08-21, branch `request_logging`. This is a real bug in
upstream vLLM 0.27.1's `mamba_cache_mode=align` implementation, not a
config problem in this repo. Current mitigation is a config change
(`--max-num-batched-tokens 1024`, see below); this doc is the writeup for
whoever chases the actual code fix later.

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
(`patches/preemption-mamba-align-freeing-debug-logging.patch`): logged
every call to this check across a full 88K-token prefill, for all 3 of
the model's Mamba/GDN KV-cache groups, at both
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
**The exact code path that makes this alignment-dependent (why 1024
reclaims and 2048 doesn't) was not traced to the line** -- the leak
mechanism (the freeing check never firing) is proven with certainty; the
modulating factor is measured with certainty; the precise null-slot-reuse
code inside `MambaManager.allocate_new_blocks()`'s align branch
(`single_type_kv_cache_manager.py`, roughly lines 1532-1600, past where
this investigation stopped reading) that decides whether a "new" mamba
block reuses a null slot vs. draws fresh is the next thing to read for
anyone chasing the actual fix.

At 2048 (the previous default), this ~2.5x-worse-than-necessary rate
exhausts the pool after ~45 steps (~75K tokens) instead of comfortably
holding the full 88K-token prompt, and the request self-preempts against
*itself* -- no other request involved.

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

## Current mitigation (applied, not a real fix)

`single-user/start_qwen.sh`: `--max-num-batched-tokens` changed from
`2048` to `1024` (old value left as a comment on the following line).
Empirically zero-waste at this workload's token counts, and no measured
throughput cost in testing (858-877 tok/s prefill either way -- this
model/GPU combo isn't compute-bound at 1024 batched tokens). This
directly contradicts `docs/gotchas.md` gotcha #7 ("2048 wins on this
card") -- that gotcha predates this investigation and needs updating,
since it was written on the assumption that the only effect of a bigger
chunk was profiled-activation-memory shrinking cache pool size, not this
leak.

**Do not** try `--max-num-batched-tokens 8192` or higher without
re-checking `--gpu-memory-utilization` headroom first -- it caused a hard
`torch.OutOfMemoryError` engine crash in testing (profiled activation
memory at that batch size left ~70MB less room than the KV cache needed
at the time; bumping `GPU_UTIL` to 0.94 worked around the startup check,
but real per-step memory pressure at 8192 still crashed the engine mid-request).

## What a real fix needs to address

Two independent things could each move the needle, and neither was
attempted live -- both touch Mamba recurrent-state correctness, not just
performance, so they need someone who knows this code path's invariants
before landing:

1. **The freeing check itself.** The comment above the check says
   `last_state_block_idx` refers to the block "allocated two steps ago"
   -- implying a *deliberate* 1-2 step safety margin, not "should never
   trigger during monotonic prefill." Naively changing the strict `<` to
   `<=` was considered and explicitly **not** done live: freeing a
   Mamba state block one step earlier than intended risks freeing state
   that's still being read from (e.g. by an in-flight copy into the next
   block), which would silently corrupt generated output rather than
   just waste memory -- a much worse failure mode than the preemption
   this investigation started from. Whether `<=` is actually safe depends
   on exactly what "two steps ago" is protecting against, which wasn't
   determined here.

2. **The chunk-size-dependent null-slot reuse.** Given the measured 1024
   vs. 2048 difference is entirely about whether the predicted per-step
   mamba draw reclaims a null slot or asks the pool for a fresh block,
   there may be a way to make that reclaim happen unconditionally
   (independent of whether chunk size happens to align with block size)
   without touching the freeing-check's timing at all. This looks like
   the safer of the two angles to pursue, since it doesn't touch when
   state actually gets freed -- only whether a "new" draw can reuse an
   already-allocated (but null-marked) slot. Starting point:
   `MambaManager.allocate_new_blocks()`'s `mamba_cache_mode == "align"`
   branch in `single_type_kv_cache_manager.py` (~line 1532 onward as of
   vLLM 0.27.1), specifically what happens after the null-block padding
   loop, past where this investigation stopped reading.

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

Diagnostic patches (apply on top of the base patch set, quiet by default
-- gated on free blocks < 15% of pool, so normal short-prompt traffic
won't spam `qwen.log`):
- `patches/preemption-per-group-alloc-debug-logging.patch` -- per-KV-group block allocation breakdown
- `patches/preemption-mamba-align-freeing-debug-logging.patch` -- the freeing-check trace
- `patches/preemption-usage-debug-logging.patch` -- raw block-pool usage numbers
- `patches/preemption-lookup-debug-logging.patch` -- prefix-cache hit sizes on preempted-request retries

grep `qwen.log` for `request-logging preemption-debug:` to find all of
the above.
