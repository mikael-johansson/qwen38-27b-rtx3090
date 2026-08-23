# Results: HIT_DIVERGED prefill recompute fix (2026-08-23)

Executed against `HIT_DIVERGED_FIX_PLAN.md` on branch `request_logging`.
Both patches are committed (`git log --oneline`: "Fix HIT_DIVERGED
reprefills: opt out of eagle-fallback for group 3" and "Rescue
HIT_DIVERGED hits the offload tier can back but the lookup can't see").

## Summary for the busy reader

- **Phase A**: Case A2 confirmed -- the MTP drafter's own attention
  layer (`mtp.layers.0.self_attn.attn`) really does share KV cache
  group 3 with the target model's 16 full-attention layers.
- **Phase B**: `VLLM_OFFLOAD_EAGLE_FALLBACK=0` (set in `micke-start.sh`)
  disables the connector's eagle mark-everything fallback entirely.
  Fixes the dominant cause (both traced events, and structurally the
  whole "turn N+1, 1-2 chunk external gap" pattern).
- **Phase C**: `patches/hit-diverged-boundary-rescue.patch` adds a
  one-shot retry for the smaller, separate boundary-blind-spot gap.
  Built and verified inert/safe; did not need to fire in the Phase D
  traffic run (Phase B alone already eliminated all divergence this
  traffic pattern produces).
- **Phase D**: 3-way replay of real captured traffic (~11 min, 3
  threads x 40 turns, comparable load to the 29-event/825K-token
  overnight baseline): **0 HIT_DIVERGED RECONCILE, 0 RESCUED**, 0
  preemptions, 0 crashes/hangs, spec-decode acceptance rate unchanged
  (~2.9-3.3 mean acceptance length vs ~2.7-3.0 pre-fix). 347,776 tokens
  restored from the offload tier across 120 hits (up to 48,256 tokens
  in a single hit) with zero full reprefills.
- **User's headline requirement** ("prefill is never recomputed once
  computed") -- met in this run: cumulative cache-hit rate reached
  98.2% (97.0% local prefix + up to 53.7% external/tier-restored of
  that) by the end of the run, zero preemptions, zero discarded KV.

## Phase A: group composition (verbatim mapping + Case A1/A2 decision)

Static trail:
- `venv/.../model_executor/models/qwen3_5_mtp.py`: `Qwen3_5MultiTokenPredictor.__init__`
  builds `self.layers` as a `ModuleList` of `Qwen3_5DecoderLayer(..., layer_type="full_attention", prefix=f"{prefix}.layers.{idx}")`
  for `idx in range(self.num_mtp_layers)` (1 layer here). `prefix` is
  `"mtp"` (set in `Qwen3_5MTP.__init__`, `maybe_prefix(prefix, "mtp")`),
  so the drafter's own attention layer registers as
  `mtp.layers.0.self_attn.attn` -- a real `full_attention`-type layer,
  not exempted from KV-cache registration.
- `venv/.../v1/worker/gpu_model_runner.py::get_kv_cache_spec()` walks
  `get_layers_from_vllm_config(vllm_config, AttentionLayerBase)` --
  i.e. every registered attention layer including the MTP module's,
  since it's a real `nn.Module` layer in the same `vllm_config`. No
  special-casing skips MTP layers.
- `venv/.../v1/core/kv_cache_utils.py::get_kv_cache_groups()` ->
  `is_kv_cache_spec_uniform()` path (all target full-attention layers +
  the one MTP attention layer share the exact same `FullAttentionSpec`,
  since there's no DeepSeek-V4-style SWA/MLA mix here) ->
  `_get_kv_cache_groups_uniform_spec()` puts them all in ONE group. The
  only place `is_eagle_group` is ever set,
  `_annotate_eagle_groups_deepseek_v4()`, only runs on the
  DeepSeek-V4-specific `group_and_unify_kv_cache_specs()` branch --
  never reached here, so group 3 is never explicitly tagged.

Runtime confirmation (TEMPORARY log line added to
`SchedulerOffloadConfig.from_spec()`, kept permanently -- it's 4 INFO
lines once per boot and was the actual instrument that resolved Case
A1 vs A2; folded into `patches/offload-eagle-misclassification-mamba.patch`):

```
KV cache group 0 spec=MambaSpec layer_names=['language_model.model.layers.0.linear_attn', ... (16 GDN layers, stride 4, idx 0,4,8,...,60)]
KV cache group 1 spec=MambaSpec layer_names=[... idx 1,5,9,...,61 ...]
KV cache group 2 spec=MambaSpec layer_names=[... idx 2,6,10,...,62 ...]
KV cache group 3 spec=FullAttentionSpec layer_names=['language_model.model.layers.3.self_attn.attn', ... (16 target full-attn layers, idx 3,7,11,...,63), 'mtp.layers.0.self_attn.attn']
```

(Full mapping in `qwen.log.evidence-mamba-free-never-fired.log:3807-3810`
and reproduced identically on every subsequent boot, e.g. `qwen.log:7082-7085`.)

**Decision: Case A2.** `mtp.layers.0.self_attn.attn` is the LAST entry
in group 3's layer list, confirming the drafter's attention KV genuinely
lives in the same cache group as the target's full-attention KV. This
overturns `docs/mamba-align-prefill-leak.md`'s assumption (stated in the
2026-08-22 patch header, itself sourced from
`single_type_kv_cache_manager.py`'s comment about the drafter having "no
mamba layers" -- true, but silent on the drafter's attention layer,
which is the actual relevant fact for the eagle guard). Per the plan,
Case A2 makes Phase D's acceptance-rate comparison mandatory gating
evidence for disabling the guard -- see the Phase D table below;
verdict: no regression, fix stands.

## Phase B: the fix

What it does (3 sentences): `SchedulerOffloadConfig.from_spec()`'s
eagle mark-everything fallback previously excluded only `MambaSpec`
groups (2026-08-22 fix), leaving group 3 marked as eagle/draft-volatile
because it's a `FullAttentionSpec` group; being marked eagle makes
`_lookup()` pop group 3's one confirmed-but-provisional trailing chunk
back to zero on every lookup, which on a "turn N+1, GPU-resident prefix,
1-chunk external gap" pattern collapses `ext` to 0 and triggers the
scheduler's HIT_DIVERGED reconcile. `VLLM_OFFLOAD_EAGLE_FALLBACK=0`
(new env var, set in `micke-start.sh`) disables the fallback entirely
instead of trying to refine its group selection further -- no group is
treated as draft-volatile, so no trailing chunk is ever popped or
excluded from storing.

Startup log line before/after:

```
# before (2026-08-22 state, both traced HIT_DIVERGED events happened under this):
KV offloading: EAGLE/MTP draft attention groups [3] detected. The trailing chunk of these groups will be excluded from offloading due to volatility.

# after:
KV offloading: EAGLE/MTP mark-everything fallback disabled via VLLM_OFFLOAD_EAGLE_FALLBACK=0; no KV cache groups are treated as draft-volatile.
```

Regression + needle results (both run twice across two separate server
boots during this session, all PASS):
- `test_offload_regression.sh`: conv A turn 2 in ~4-5s both times;
  `qwen.log` shows `hit 87360 offloaded tokens after 0 GPU hit tokens`
  both times (full 87,360-token prompt served entirely from the
  offload tier, zero recompute).
- `needle_recall_test.py -p 4000 --turns 2`: 6-digit code embedded
  ~2/3 into a ~4,000-sentence filler prompt, turn 2 (resuming from
  offloaded state, temperature 0) recalled the code exactly both runs
  (`164742`, `271299`). Turn 2 took 2.4-2.5s.

## Phase C: the boundary rescue

Diff summary: `v1/core/sched/scheduler.py`'s
`if hit_diverged and num_external_computed_tokens == 0:` branch gets a
retry before falling through to reconcile -- re-query
`connector.get_num_new_matched_tokens(request, retry_local)` where
`retry_local = round_down(block_aligned_local, align_size) - align_size`
(`align_size` from a new `get_mamba_align_size()` accessor added to
`OffloadingConnectorScheduler`/`OffloadingConnector`, looked up via
`getattr` so non-offloading connectors are unaffected). `ext2 > 0`:
truncate the local computed blocks to `retry_local`
(`kv_cache_manager.truncate_computed_blocks`, the same helper the
existing "remote strictly exceeds local hit" path already uses) and
resume with `num_external_computed_tokens = ext2`, logged
`HIT_DIVERGED RESCUED`. `ext2 is None`: requeue the request
(`step_skipped_waiting.prepend_request`), same as the connector's
normal deferred-lookup path. `ext2 == 0`: fall through to the existing
`HIT_DIVERGED RECONCILE` unchanged.

Invariants (§6 of the plan), one line each:
1. Never resume without a confirmed-loadable mamba chunk at the
   boundary -- verified by reading `update_state_after_alloc`: it
   early-returns on `num_external_tokens == 0`, and the rescue only
   sets `num_external_computed_tokens = ext2` when `ext2 > 0`, so the
   normal load-job path (which does the confirming) is always the one
   that runs.
2. `get_num_new_matched_tokens` is safe to repeat -- read: it clears
   `group_state.block_ids` and re-sets `num_locally_computed_tokens` at
   entry (top of the function), and separately checks
   `req_status.transfer_jobs` for in-flight transfers at entry, so a
   real in-flight transfer would have made the *first* call in the
   branch return `None` already, meaning the retry can never race a
   transfer the first call itself kicked off.
3. At most one retry per schedule attempt -- straight-line code (no
   loop); confirmed by reading the diff, there is exactly one call to
   `get_num_new_matched_tokens` inside the rescue block.
4. Async-scheduling / `build_connector_meta()` boundary respected --
   the entire rescue lives inside the existing `schedule()` lookup
   phase, before `allocate_slots`/`update_state_after_alloc`; nothing
   was moved.

Repro script behavior (`test_hit_diverged_rescue.sh`): single
conversation, `-p 2000` first turn, short reply, second turn with no
intervening eviction pressure. Ran clean: `turn 2 done: 08:32:14 (4s)`,
`RESCUED=0 RECONCILE=0` -- **no divergence occurred at all** in this
solo run (expected/degrades-gracefully per the plan -- provoking the
exact shallow-mamba/deep-attention split needs real concurrent
contention, which the Phase D multi-thread run below provides instead).
The script's own PASS/FAIL logic (0 RECONCILE + turn-2 time under 30s)
reported PASS.

**The rescue path itself did not fire during Phase D either** (0
RESCUED alongside 0 RECONCILE) -- Phase B's fix already eliminates
100% of the divergence this traffic pattern produces, so Phase C is
verified safe-and-inert rather than verified-firing-correctly-under-
load. Confirming an actual `HIT_DIVERGED RESCUED` firing would need
either the exact `num_computed_tokens` boundary landing precisely
between two align-chunk multiples under real contention (rare enough
that neither of the two originally-traced events needed it -- both were
Case-A2-eagle-pop, not boundary-blind-spot) or a targeted synthetic
repro that pins `num_computed_tokens` there directly. Left as an open
question below rather than forced.

## Phase D: measurement matrix

Only one live configuration was run this session (see rationale below
for why Phase B and B+C are reported together, and why no fresh
"Phase B only" run was collected).

| metric | Baseline (pre-fix, overnight, per plan doc) | Baseline (pre-fix, evidence-log sample) | Phase B+C (this session, 3-way, ~11 min, 40 turns/thread) |
|---|---|---|---|
| HIT_DIVERGED RECONCILE events / tokens discarded | 29 events / 825,000 tokens (20 min, 3-way; recorded number, not re-derived -- the source log no longer exists as a file) | 2 events / 43,264 tokens (excerpt in `qwen.log.evidence-mamba-free-never-fired.log`) | **0 events / 0 tokens** |
| HIT_DIVERGED RESCUED events / tokens saved | n/a (patch didn't exist) | n/a | 0 (Phase B left nothing for it to rescue in this run) |
| offload hits | not recorded | not recorded | 120 hits, 347,776 tokens total, largest single hit 48,256 tokens |
| preemptions | not recorded | not recorded | 0 (`num_preemptions=0` on every retried-request log line; 0 actual preemption-retry events) |
| peak GPU KV cache usage | not recorded | not recorded | 90.2% |
| cumulative cache-hit rate (end of run) | not recorded | not recorded | prefix 97.0%, external (of prefix) up to 53.7%, total 98.2% |
| spec-decode mean acceptance length | not recorded | ~2.71-3.04 (sample of 5 `SpecDecoding metrics` lines) | ~2.91-3.26 (sample of 5 lines); aggregated over all 5,180 generation-log entries: 2.967/4.000 accepted/verified = 74.2% (vs 73.6% over 1,683 entries in the evidence-log sample) |
| hangs / AcceleratorError | not recorded | 0 | 0 (`grep -ci "AcceleratorError\|illegal memory access"` = 0; the four `Avg prompt throughput: 0.0` lines are normal idle/single-request heartbeats, not frozen-loop signatures -- checked `Running`/`GPU KV cache usage` values, they vary normally around them) |

**Configuration actually run, and why it doubles as both "Phase B only"
and "Phase B + C"**: this session ran ONE live 3-way traffic
replay with both patches applied together (not a separate Phase-B-only
run per the plan's config 2). Since the run produced **zero
HIT_DIVERGED events of any kind** (0 RECONCILE, 0 RESCUED), Phase C's
code was never exercised -- every decode iteration, every offload hit,
and the entire measured acceptance rate in this run are consequences of
Phase B alone. The acceptance-rate numbers in the table above are
therefore valid gating evidence for the Case A2 decision (disabling the
eagle guard on group 3) without needing a separate isolated run.
Rationale for not also running a literal Phase-B-only config: GPU-hours
(the plan says to prefer reusing the recorded baseline for the same
reason), and the logical argument above already isolates the effect.

**Config 4 (headline scenario)** -- the 3-way replay of real ~150-180
message-deep Hermes conversations (prompt sizes growing to 42K/66K/74K
tokens by turn 40 across the three threads) against a 171,956-token GPU
pool is a reasonable stand-in for "2+ concurrent large conversations
that don't all fit in VRAM sustained": peak GPU KV usage hit 90.2%,
meaning real contention/eviction pressure occurred, yet zero
preemptions and zero discarded KV resulted. Per-turn `Cached tokens`
percentage from `requests/*.log` was not separately extracted this
session (the `prefix cache hit` / `external` telemetry in `qwen.log`,
reproduced in the table above, is the equivalent aggregate signal and
was judged sufficient); flagged as a follow-up if per-turn granularity
is wanted.

Representative log excerpt (the two original divergence events, for
reference -- neither reproduced in the Phase D run since the underlying
bug is fixed):

```
# qwen.log.evidence-mamba-free-never-fired.log:27848-27855 (pre-fix)
group 3 _maximal_prefix_lookup start_chunk_idx=48 scanned=17 hit_count=1 defer_lookup=True stop_reason=exhausted
_sliding_window_lookup len(keys)=17 sliding_window_size=1 scanned=17 consecutive_hits=1 any_uncertain=True run_has_pending=False result=1
_sliding_window_lookup len(keys)=1 sliding_window_size=1 scanned=1 consecutive_hits=1 any_uncertain=False run_has_pending=False result=1
_sliding_window_lookup len(keys)=1 sliding_window_size=1 scanned=1 consecutive_hits=1 any_uncertain=False run_has_pending=False result=1
group 3 _maximal_prefix_lookup start_chunk_idx=48 scanned=1 hit_count=1 defer_lookup=False stop_reason=exhausted
HIT_DIVERGED RECONCILE -- diverged local hit was 39936 tokens, external connector confirmed 0, reconciled down to 0 tokens (the boundary every KV group agrees on) -- 39936 tokens of otherwise-valid hit discarded for Mamba-state safety
```

```
# qwen.log.evidence-mamba-free-never-fired.log:5109 (pre-fix, smaller event, same mechanism)
HIT_DIVERGED RECONCILE -- diverged local hit was 3328 tokens, external connector confirmed 0, reconciled down to 0 tokens (the boundary every KV group agrees on) -- 3328 tokens of otherwise-valid hit discarded for Mamba-state safety
```

## Anomalies

None encountered this session:
- No hangs (no `VLLM::EngineCore` pinned, no frozen `GPU KV cache
  usage`/`Avg prompt throughput: 0.0` alongside nonzero `Running`).
- No `torch.AcceleratorError` / illegal memory access (the known rare
  crash from the plan's ground rules did not recur).
- No acceptance-rate regression (see table).
- No throughput regression observed qualitatively (per-turn `ttfb`/
  `total` times in the simulator output stayed in the low single-digit-
  to-low-double-digit seconds range throughout the 3-way run, consistent
  with prior sessions' numbers for this traffic pattern).
- Server restarted cleanly multiple times this session (three boots:
  Phase A verification, Phase B verification, Phase B+C verification +
  Phase D run) with no leftover-process or GPU-memory issues, confirmed
  via `nvidia-smi`/`ps` before each restart.

## Open questions

1. **Phase C's rescue path never fired under load in this session's
   testing** -- built, code-reviewed against all 4 invariants, and
   confirmed inert/harmless (0 RECONCILE, 0 RESCUED, no regressions),
   but not confirmed *firing correctly* against a real boundary-blind-
   spot case. What was tried: the solo repro script
   (`test_hit_diverged_rescue.sh`, degraded gracefully to "no
   divergence" as the plan anticipated) and the 3-way traffic replay
   (also 0 divergence of any kind). Next step if this needs closing:
   either a synthetic repro that pins `num_computed_tokens` to land
   exactly between two 832-token align-chunk boundaries (would need a
   small harness poking the scheduler's internal state directly, more
   invasive than anything built this session), or just leave it as
   defense-in-depth and revisit if a real `HIT_DIVERGED RECONCILE`
   ever reappears in production logs (grep for it -- if `RECONCILE`
   shows up with 0 corresponding `RESCUED` nearby, that's the signal
   Phase C isn't covering that specific case and needs a closer look).
2. **No literal "Phase B only" run** was collected as its own config
   (see the Phase D table's rationale note) -- if the user wants a
   strictly isolated number rather than the logical argument given,
   it's a ~10-15 min rerun with `patches/hit-diverged-boundary-rescue.patch`
   reverted (`patch -p1 -R -d $SP < patches/hit-diverged-boundary-rescue.patch`,
   restart, rerun `simulate_hermes_traffic.py --threads 3 --max-turns 40`,
   re-apply the patch afterward).
3. **The 29-event/825K-token overnight baseline** is cited from the
   plan document's own text; the underlying log file no longer exists
   in the repo (only the smaller `qwen.log.evidence-mamba-free-never-
   fired.log` excerpt survives, showing 2 events / 43,264 tokens over
   its shorter capture window). Both are consistent with the same root
   cause and both are now zero after the fix; flagging only so the
   829K-token figure in the summary above is understood as a carried-
   forward citation, not something this session re-measured.
4. **Per-turn `Cached tokens` percentage / prefill duration** from
   `requests/*.log` (plan's exact ask for config 4) was not separately
   extracted -- the aggregate `qwen.log` cache-hit telemetry was used
   instead as a faster proxy. If per-turn granularity matters (e.g. to
   confirm literally every turn, not just the aggregate, avoided
   reprefill), it's derivable from the `requests/*.log` files already
   captured during the Phase D run (timestamps ~08:32-08:43 UTC
   2026-08-23) plus the corresponding `qwen.log` prompt-processing
   lines in the same window.

## Housekeeping

- `patches/offload-verbose-eviction-debug-logging.patch` is **still
  applied** (per the plan's instruction, left for the user's live-
  traffic observation). Revert once this investigation closes -- it is
  the source of every `request-logging offload-verbose:` line used
  throughout this document and is fairly chatty at DEBUG level.
- `bash verify.sh --no-server` and `bash verify.sh` (server up) both
  run clean except the 4 pre-existing DFlash2/spec-decode-attn
  failures (`dflash2-backport.patch`, `dflash2-lookup-drafting.patch`,
  `hybrid-kv-groups-v2-cudagraph.patch`, `spec-decode-attn.patch`) --
  unrelated to this work, not installed/applicable in this environment,
  present before this session started.
- All code from this session (both patches, `micke-start.sh`, the two
  test scripts, `needle_recall_test.py`) is already committed on
  `request_logging` (2 commits: "Fix HIT_DIVERGED reprefills..." and
  "Rescue HIT_DIVERGED hits..."). This results file and the
  `docs/mamba-align-prefill-leak.md` addendum are the remaining
  Phase E deliverables, committed together as the final commit of this
  work.
