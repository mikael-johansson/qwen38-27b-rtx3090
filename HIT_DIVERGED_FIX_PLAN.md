# Plan: eliminate HIT_DIVERGED prefill recompute (eagle misclassification, part 2 + boundary rescue)

> **For the executing agent (Sonnet):** this file is self-contained. Read it fully
> before touching anything. The background investigation lives in
> `docs/mamba-align-prefill-leak.md` (long; you do NOT need to re-read all of it —
> the relevant, *corrected* conclusions are restated below, and one of that doc's
> final conclusions is overturned by new evidence in this plan). Work on branch
> `request_logging`. When done, write your results into `RESULTS-hit-diverged.md`
> (format specified at the bottom) for the user to relay back for analysis.

## 0. Context you must know before starting

**Deployment**: single RTX 3090, vLLM 0.27.1 in `venv/` (patched — see
`patches/*.patch`, applied into
`venv/lib/python3.12/site-packages/vllm/`, referred to as `$SP` below).
Model: Qwen3.8-27B (hybrid GDN/linear-attention + full attention,
architecture `Qwen3_5MTP`), MTP speculative decoding (`method: mtp`,
3 spec tokens), `--mamba-cache-mode align`, prefix caching on, KV
offloading via `OffloadingConnector`/`TieringOffloadingSpec` (20 GB CPU
tier + 200 GB NVMe fs tier). Started with `bash micke-start.sh`; log is
`qwen.log`; API on `127.0.0.1:18020` (key in `api_key.txt`).

**The model has 4 KV-cache groups sharing one block pool**: groups 0–2 are
`MambaSpec` (GDN recurrent state, checkpointed at 832-token block
boundaries under align mode), group 3 is full attention. All groups are
offloaded/restored by the connector; keys are content-hash based
(`PYTHONHASHSEED=0` keeps them stable across restarts).

**The goal** (user's requirement): once prefill has been computed for a
long conversation, it must NEVER be recomputed — swap to RAM/NVMe and
restore instead. Sub-block tails (≤832 tokens) recomputed per turn are
acceptable.

**Where we are**: the mamba-align leak, the mamba-group eagle
misclassification, the flush-protect eviction race, and the fs-tier disk
cap are all already fixed (patches applied — do not touch them). The
dominant *remaining* waste is `HIT_DIVERGED RECONCILE` events: the
scheduler finds a deep local (GPU) full-attention prefix hit, the mamba
groups' local hit is shallower, the offload connector reports 0 external
tokens, and the scheduler — correctly, for mamba-state safety — throws
away the ENTIRE deep hit and reprefills from the shallow common boundary.
Measured overnight 2026-08-23: 29 events / 825K tokens discarded in one
20-minute 3-way run.

## 1. The corrected root cause (new evidence — this overturns the doc's last conclusion)

`docs/mamba-align-prefill-leak.md`'s final section concluded the
HIT_DIVERGED losses were content that "never had a store job queued" and
recommended a large block_pool↔connector coupling ("synchronous store on
eviction"). **Direct trace analysis of
`qwen.log.evidence-mamba-free-never-fired.log` shows that conclusion is
wrong for both captured events.** Do NOT build the synchronous-store
mechanism; it is explicitly out of scope.

Both HIT_DIVERGED events in that log (lines ~5102–5109 and ~27848–27855)
have this signature — e.g. the second (request `...-9f40098d`,
54,794-token prompt):

```
group 3 _maximal_prefix_lookup start_chunk_idx=48 scanned=17 hit_count=1 ... stop_reason=exhausted
_sliding_window_lookup len(keys)=17 sliding_window_size=1 ... consecutive_hits=1 ... result=1
_sliding_window_lookup len(keys)=1  sliding_window_size=1 ... consecutive_hits=1 ... result=1   (x2, mamba groups)
group 3 _maximal_prefix_lookup start_chunk_idx=48 scanned=1 hit_count=1 defer_lookup=False
HIT_DIVERGED RECONCILE -- diverged local hit was 39936 tokens, external connector confirmed 0, ...
```

Read that carefully: the local attention hit was 48 chunks (39,936
tokens); the offload tier was scanned beyond chunk 48 and **every mamba
group returned a confirmed HIT at chunk 48, and attention externally hit
chunk 48 too** (1 chunk, then `exhausted` — chunks 49+ are the genuinely
new user message, correctly absent). The tier could fully back a resume
at 40,768 tokens. Yet ext came back 0 and 39,936 tokens were discarded.

**Why ext became 0**: group 3 (full attention) is still classified as an
EAGLE/MTP draft group. The earlier fix
(`patches/offload-eagle-misclassification-mamba.patch`) excluded only
`MambaSpec` groups from the fallback in
`SchedulerOffloadConfig.from_spec()`
(`$SP/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`,
search for `use_eagle and not eagle_groups`). Startup log confirms:

```
KV offloading: EAGLE/MTP draft attention groups [3] detected. The trailing chunk ...
```

In `_lookup()` (same file, search for `is_eagle_unverified`), an eagle
group pops one provisional trailing chunk after a successful lookup:
`num_hit_chunks -= 1`. The attention group's external hit was exactly 1
chunk → popped to 0 → `max_hit_size_tokens` collapses to the local
boundary → `new_num_hit_tokens < tokens_per_chunk` → `return 0` → the
scheduler's reconcile fires and discards the whole deep hit.

This fires on precisely the production pattern (turn N+1 of a
conversation whose shared prefix is still mostly GPU-resident: the
external hit beyond the local boundary is almost always exactly 1–2
chunks — the previous turn's decode tail). An 832-token conservatism gets
amplified into a 30–50K-token recompute.

**Secondary effect of the same misclassification (store side)**: in
`storable_chunks()` (same file, `is_eagle_group and is_decoding` branch),
the newest complete chunk of an eagle group is excluded from storing
during decode — so each finished turn's final complete attention chunk is
never stored, weakening exactly the chunk the next turn needs.

**Why the eagle guard exists at all**: for groups genuinely holding
EAGLE/MTP *draft-model* KV, the trailing chunk's draft KV can be
rewritten after spec-token rejection, so a stored copy may be stale.
Upstream tags such groups per-architecture (`is_eagle_group` is only ever
set by a DeepSeek-V4-specific hack in `$SP/v1/core/kv_cache_utils.py`,
`_annotate_eagle_groups_deepseek_v4`). For this model nothing is tagged,
so the connector's "mark everything" fallback fires.
`docs/mamba-align-prefill-leak.md` asserts none of the 4 groups hold the
MTP draft head's own state — Phase A verifies that claim.

**Even in the worst case** (drafter KV really does share group 3), a
stale draft-KV chunk can only lower the spec-decode *acceptance rate*
briefly near a resume boundary — MTP drafts are always verified by the
target model, so output correctness is unaffected. This bounds the risk
of removing the guard.

## 2. Second, smaller structural gap (Phase C)

`_lookup()` starts every group's scan at
`start_chunk_idx = num_computed_tokens // tokens_per_chunk` — strictly
*beyond* the presented local boundary. A mamba checkpoint sitting exactly
AT that boundary (chunk `start_chunk_idx - 1`, e.g. when the previous
turn's decode never crossed the next 832 boundary) is invisible. Then
ext is legitimately 0 by API semantics ("N NEW tokens"), and the
reconcile discards a deep hit the tier could in fact back. This will keep
producing occasional large discards even after Phase B.

## 3. Ground rules (read carefully)

- **Branch**: stay on `request_logging`. Commit logically-separate
  changes separately, with descriptive messages. Never commit to `main`.
- **Patch workflow** (matches every existing patch): edit the live file
  under `$SP`, then produce a unified diff with git-style `a/`,`b/` paths
  **relative to `$SP`** (e.g.
  `--- a/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`)
  into `patches/<name>.patch`, with a prose header at the top of the
  patch file explaining the why (see
  `patches/offload-eagle-misclassification-mamba.patch` for the expected
  header style — root cause, evidence, what the fix does, how verified).
  `bash verify.sh --no-server` must report every patch as applied
  (it does `patch -p1 -R --dry-run -d $SP`). If you modify behavior
  covered by an existing patch, **regenerate that patch file** (do not
  stack a conflicting second patch on the same hunks) and say so in its
  header changelog.
- **Line numbers drift** (debug patches are applied): locate code by
  symbol/grep, never by the line numbers quoted here.
- **Server restarts**: the server may be serving the user's real agent
  traffic. Before killing/restarting it, check for recent activity
  (`ls -lt requests/ | head`; anything modified in the last ~10 minutes
  means live traffic) and if in doubt, stop and ask the user. Start with
  `bash micke-start.sh`, wait for `curl -sf 127.0.0.1:18020/health`.
- **If the engine ever hangs** (heartbeat lines in `qwen.log` with
  `Avg prompt throughput: 0.0`, frozen `GPU KV cache usage`,
  `VLLM::EngineCore` pinned at 100% CPU): get the EngineCore PID
  (`ps aux | grep VLLM::EngineCore`), `kill -USR1 <pid>`, save
  `/tmp/vllm_faulthandler_dump.txt`, then kill the server. Record the
  dump in your results file — this is high-value evidence.
- **Known rare crash, not yours to fix**: `torch.AcceleratorError: CUDA
  error: an illegal memory access` under 3-way load (seen once,
  2026-08-23). If it happens, save the traceback from `qwen.log` into
  your results, restart, continue. Do not chase it.
- **Out of scope — do not build**: the synchronous
  store-before-eviction block_pool↔connector coupling; any change to
  `patches/mamba-align-stale-state-queue.patch`; any loosening of the
  HIT_DIVERGED reconcile that would resume without confirmed mamba state
  at the resume boundary (that is a silent-corruption risk, never
  acceptable).
- Keep `patches/offload-verbose-eviction-debug-logging.patch` applied
  throughout — its `request-logging offload-verbose:` lines are your
  measurement instrument. `micke-start.sh` already points
  `VLLM_LOGGING_CONFIG_PATH` at the debug logging config.

## 4. Phase A — verify group 3's composition (blocks Phase B's shape)

**Question**: does any MTP-drafter layer's KV live in KV-cache group 3,
or is the eagle fallback a pure false positive for this model?

1. Static: read the model implementation (grep `$SP/model_executor/models/`
   for the `Qwen3_5MTP` architecture / `qwen3_5_mtp.py`) and how drafter
   layers register KV caches (does the MTP draft block contain an
   attention layer that allocates a KV cache in the main
   `kv_cache_spec`? how are its layer names prefixed — e.g. `mtp.`?).
   Also check how `get_kv_cache_groups()` in
   `$SP/v1/core/kv_cache_utils.py` groups them.
2. Runtime confirmation: add a TEMPORARY log line in
   `SchedulerOffloadConfig.from_spec()` printing, for each group:
   `group_idx`, spec class name, and `kv_cache_config.kv_cache_groups[idx].layer_names`.
   Restart the server once and capture the output. Remove the temporary
   line afterwards (or fold it into the verbose-debug patch if it seems
   permanently useful).
3. Record the verbatim group→layer_names mapping in your results file.

**Decision**:
- **Case A1 — no drafter layers in any group** (expected, per the doc's
  claim): proceed with Phase B as written.
- **Case A2 — drafter layers ARE in group 3**: still proceed with Phase
  B (the correctness argument in §1 holds — verification bounds the risk
  to acceptance rate), but Phase D's acceptance-rate comparison becomes
  mandatory gating evidence, and your results file must flag this
  loudly.

## 5. Phase B — fix the eagle classification for group 3

**Goal**: group 3 must stop being treated as an eagle group (lookup pop
AND store-side trailing-chunk exclusion both disappear).

Implementation (regenerate
`patches/offload-eagle-misclassification-mamba.patch` — same file, same
fallback block in `from_spec()`):

- Case A1: the fallback should produce an **empty** eagle set when no
  group actually contains a draft-model layer. Preferred: detect
  drafter-layer membership using whatever reliable signal Phase A found
  (e.g. layer-name prefix from the speculative config's draft model);
  derive it from config, do not hardcode the string `"mtp."` without
  checking what the names actually are. If no robust in-code signal
  exists, acceptable fallback: a new env var (e.g.
  `VLLM_OFFLOAD_EAGLE_FALLBACK=0` to disable the mark-everything
  fallback; default `1` = unchanged upstream behavior), set to `0` in
  `micke-start.sh` with a comment block explaining why (match the
  existing comment style in that script). Either way the patch header
  must document the new evidence from §1.
- Case A2: same mechanics, but the header must state the acceptance-rate
  tradeoff and Phase D gates the decision.

Update the startup log line accordingly (it currently prints
`EAGLE/MTP draft attention groups [3] detected` — after the fix it must
print an empty set / not print, and that's your smoke-test signal).

**Immediately verify** (before Phase C):
1. `bash verify.sh --no-server` → all patches PASS.
2. Restart server; startup log no longer marks group 3 as eagle.
3. `bash test_offload_regression.sh` → PASS (conv A turn 2 in ~5s with a
   `hit N offloaded tokens` line). This guards against regressing the
   original mamba-group fix.
4. Correctness spot-check (needle recall): embed a unique 6-digit code
   deep inside a long prompt via `local_chat_test.py -p 4000` style
   usage, run two turns so turn 2 resumes from offloaded state, and ask
   for the code back with temperature 0. It must be recalled exactly.
   (Precedent: this exact check was used to validate the leak fix — see
   the doc. Look at `local_chat_test.py --help` for how to drive it; if
   it can't embed a needle directly, write a tiny variant script in the
   repo root.)
5. Commit (venv change is not tracked; the regenerated patch + any
   micke-start.sh change is).

## 6. Phase C — rescue the deep hit when ext == 0 (boundary blind spot)

**Goal**: in the scheduler's reconcile branch
(`$SP/v1/core/sched/scheduler.py`, search for `HIT_DIVERGED RECONCILE`),
before giving up and reconciling down, give the connector one more chance
to back the deep boundary using content it can only see with a lower
presented boundary.

**Approach (option B from the analysis — deliberately the simple one)**:
in the `if hit_diverged and num_external_computed_tokens == 0:` branch,
re-query the connector once with the presented local hit lowered by one
chunk:

```python
retry_local = block_aligned_local - tokens_per_chunk   # one mamba-align chunk back
ext2, load_kv_async2 = self.connector.get_num_new_matched_tokens(request, retry_local)
```

- `tokens_per_chunk` here is the mamba-align chunk size (832 for this
  model). Get it from the connector/config rather than hardcoding
  (`resolve_mamba_align_size` / the connector's group configs — expose it
  cleanly if needed).
- If `ext2` is a positive int: use `retry_local` as the local boundary
  and `ext2` as the external tokens (i.e. resume at
  `retry_local + ext2` ≥ the old deep boundary minus one chunk, usually
  equal to the old deep boundary). You must also present the
  correspondingly TRUNCATED local computed blocks
  (`truncate_computed_blocks(new_computed_blocks, retry_local)` — the
  same helper the partial-tail path above already uses) so the last
  local chunk is re-loaded from the tier instead of adopted locally —
  this is what makes the mamba group's boundary chunk loadable at all,
  and it costs one extra attention-chunk DMA (megabytes — trivially
  cheaper than a 30–50K-token recompute).
- If `ext2` is `None` (connector deferred): treat like the existing
  deferred path (requeue the request; do NOT reconcile-and-recompute).
- If `ext2 == 0`: fall through to the existing reconcile exactly as
  today (the tier genuinely can't back the boundary; recompute is
  correct).
- Log the outcome distinctly:
  `HIT_DIVERGED RESCUED -- deep hit N kept, resumed at M via retry-lookup`
  vs the existing `HIT_DIVERGED RECONCILE` line — Phase D counts both.

**Correctness invariants you must preserve** (check each against the
code, they are why this branch is delicate):
1. Never resume at a boundary without a confirmed-loadable mamba chunk
   whose end is exactly that boundary. (The connector's `_lookup` +
   `update_state_after_alloc` machinery already guarantees this when
   ext > 0 — that's why option B reuses it instead of adding a flag/side
   channel. Note `update_state_after_alloc` early-returns when
   `num_external_tokens == 0`; do not try to make it load with ext==0.)
2. The second `get_num_new_matched_tokens` call must be safe to repeat:
   it clears per-group `block_ids` and re-sets
   `num_locally_computed_tokens` at entry (verify this by reading it —
   it looked re-entrant in analysis, confirm).
3. Respect the existing `ext_tokens is None` ("connector needs more
   time") and in-flight-transfer semantics — never busy-loop the retry
   (at most ONE retry lookup per schedule attempt).
4. Async-scheduling is on (`--async-scheduling`); do not move the
   reconcile logic across the `build_connector_meta()` boundary.

Ship as a new patch, e.g.
`patches/hit-diverged-boundary-rescue.patch`, with the full header
writeup. Then: `verify.sh --no-server`, restart,
`test_offload_regression.sh`, needle check — all must pass.

**A unit-style repro for the rescued path** (build it, it's the core
deliverable of this phase): a script like `test_hit_diverged_rescue.sh`
that provokes the exact pattern: conversation turn 1 (long prompt, e.g.
`local_chat_test.py -p 2000`), short reply, then turn 2 while the GPU
pool still holds the attention chain (do NOT evict with a second big
prompt — the point is a deep LOCAL attention hit with a shallower local
mamba hit). Success = `HIT_DIVERGED RESCUED` (or no divergence at all)
and a turn-2 TTFT of seconds, never a full reprefill. If you cannot
reliably provoke divergence solo, degrade gracefully: run two
interleaved medium conversations (see `simulate_hermes_traffic.py
--threads 2`) and assert zero `RECONCILE` events with nonzero `RESCUED`
events over the run.

## 7. Phase D — measurement matrix (the numbers the user wants back)

All runs against `bash micke-start.sh`, using
`venv/bin/python simulate_hermes_traffic.py` (replays real captured
conversations from `requests/*.log`; `--threads N`, `--max-turns N`).
The overnight baseline was ~20 minutes, 3 threads. For each
configuration record, from `qwen.log`:

| metric | how |
|---|---|
| HIT_DIVERGED RECONCILE events + tokens discarded | grep `HIT_DIVERGED RECONCILE`, sum the "discarded" numbers |
| HIT_DIVERGED RESCUED events + tokens saved | grep `HIT_DIVERGED RESCUED` (Phase C onward) |
| offload hits | grep `hit .* offloaded tokens after` |
| preemptions | grep `num_preemptions` / `Preempt` |
| aggregate + per-request throughput, TTFT per turn | the `prompt processing`/`loggers.py` lines + `requests/*.log` timings |
| spec-decode acceptance | the accepted-tokens-per-iteration log lines (ITERATION_LOG=1 is already on) — compare mean accepted length |
| hangs / AcceleratorError | grep; must be zero (crash: see ground rules) |

Configurations to run (fresh server restart between each; note the NVMe
tier persists across restarts by design — that's fine, it matches
production, just keep it consistent across runs):
1. **Baseline** (pre-Phase-B code) — only if not already derivable from
   the existing overnight numbers (29 events / 825K tokens, 20 min,
   3-way); prefer reusing the recorded baseline to save GPU-hours.
2. **Phase B only** — expectation: RECONCILE events collapse for
   turn-continuation traffic; acceptance rate unchanged (gating check
   for Case A2).
3. **Phase B + C** — expectation: remaining RECONCILE ≈ only
   genuinely-cold content (fresh server + content never computed);
   RESCUED > 0; zero correctness anomalies.
4. **The user's headline scenario**: 2 concurrent ~80K conversations
   that do NOT both fit in VRAM sustained (3-way if needed to force
   swap-out), multi-turn. Success criterion — the user's actual goal:
   **no turn ever re-prefills content that was previously computed**,
   i.e. every turn's prefill work ≈ (new tokens this turn + ≤1–2 chunks
   of tail), served from GPU cache or RAM/NVMe restore. Report each
   turn's `Cached tokens` percentage / prefill duration from
   `requests/*.log`.

## 8. Phase E — wrap up

1. Update `docs/mamba-align-prefill-leak.md`: add a dated section
   correcting the "never had a store job queued" conclusion (cite the
   two traced events + the eagle-pop mechanism), describing both fixes,
   and marking the synchronous-store idea as "not needed on current
   evidence". Keep the existing text intact (the doc's convention is
   append-and-correct, never rewrite history).
2. Leave the verbose-debug patch applied (the user will run live traffic
   and report back), but note in the results file that it should be
   reverted once the investigation closes.
3. Commit everything; `bash verify.sh --no-server` must be fully green.

## 9. Results file format (`RESULTS-hit-diverged.md`)

Structure it so a follow-up analysis session can act on it without
re-deriving anything:

- **Phase A**: verbatim group→layer_names mapping; Case A1 or A2; the
  static-analysis trail (files read, what registers drafter KV where).
- **Phase B**: what the fix does (3 sentences); startup-log line
  before/after; regression + needle results.
- **Phase C**: the actual diff summary; how the invariants in §6 were
  each verified (one line each); repro script behavior.
- **Phase D**: the full metric table per configuration, plus 2–3
  verbatim log excerpts per interesting event (especially: any remaining
  RECONCILE event — include the surrounding `_maximal_prefix_lookup` /
  `_sliding_window_lookup` lines exactly like §1's excerpt, so the
  mechanism is attributable).
- **Anomalies**: anything unexpected — hangs (with faulthandler dump),
  crashes, acceptance-rate shifts, throughput regressions, weird lookup
  patterns. Raw evidence over interpretation: when in doubt, paste the
  log lines.
- **Open questions** you couldn't resolve, each with what you tried.
