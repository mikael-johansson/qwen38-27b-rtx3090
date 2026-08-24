# Results: KV-offload restore corruption root-cause investigation

Executed per `RESTORE_CORRUPTION_PLAN.md`. Branch `restore-corruption` (off
`request_logging`).

## TL;DR

**Root cause NOT found.** What Phase C rules out is as important as what it
doesn't: for both a clean and a corrupted restore, the Mamba/GDN running-state
bytes are **byte-identical** across store→(RAM residency)→load→GPU-read (all
three groups, all layers checked) — so the specific chunk a restored request
resumes from is neither corrupted in RAM nor mis-transferred nor
mis-addressed on the GPU. Whatever produces token soup is not "wrong bytes at
this checkpoint." The leading structural hypothesis (offload racing the
Mamba/GDN checkpoint's own same-request freeing,
`mamba-align-stale-checkpoint-offload-slack.patch`'s exact mechanism) was
tested directly (`VLLM_MAMBA_ALIGN_STALE_SLACK_BLOCKS=2`) against **output
coherence** specifically (that patch was previously only ever measured against
HIT_DIVERGED counts) — **no effect**, same corruption rate. `VLLM_OFFLOAD_
EAGLE_FALLBACK` stays at its current default (`=0`, fallback disabled) per
Phase A — the corruption reproduces identically with it either way, so this
knob is unrelated and the production default is left unchanged. See
Anomalies for the sharpest remaining leads: the attention group (index 3) was
never checksummed on the GPU-read side (only Mamba groups 0-2 were — Phase C
covers 4/4 groups store→load, but only 3/4 groups load→GPU-read), and local
prefix-cache reuse for identical content stops one full chunk short of where
offload restore resumes from (57,408 vs 58,240 tokens) — unexplained, and the
next thing to chase.

## Phase A — the Phase-B revert test

**Setup:** `VLLM_OFFLOAD_EAGLE_FALLBACK=0` removed from `micke-start.sh`
(commenting mid backslash-continuation silently breaks the whole line chain —
verified with a standalone repro before touching the real script; deleted the
line instead). fs cache dir moved aside to
`/d/nvme_cache/vllm_kv_stashed/vllm_kv.pre-phaseA.<ts>` (not deleted), fresh
`/d/nvme_cache/vllm_kv` created. Server restarted; startup log confirmed the
fallback is active again: `KV offloading: EAGLE/MTP draft attention groups
[3] detected` (`qwen.log`, 09:59:42).

**Exposure problem with the plan's default invocation:** this server's GPU KV
cache pool is 171,956 tokens (`GPU KV cache size: 171,956 tokens`, startup
log) — not the ~6.11 GiB pool the original doc's repro assumed. Two ~58K-token
conversations (116K tokens) mostly both fit resident, so the default
`--convs 2` invocation gives unreliable exposure:

- Run 1 (default, `qwen.log.2026-08-24-phaseA-run1`): 8/8 coherent, but only
  **one** real restore happened (`Request chatcmpl-adf96041c7bf56b7-82a70168
  hit 57408 offloaded tokens after 0 GPU hit tokens`, 10:02:06) — turn r2/c0,
  reply `'ok'`, clean.
- Run 2 (default, immediately after, same server, `qwen.log.2026-08-24-phaseA-run2`):
  0 restores (`GPU had 0 tok` count = 0) — both conversations had settled
  resident. **Discarded as non-exposed per the ground rules.**

Switched to `--convs 4 --rounds 4` (4×58K ≈ 232K > 171,956 pool, forces real
eviction every round) to get reliable exposure matching the doc's original
repro conditions:

- Run 3 (`--convs 4 --rounds 4`, `qwen.log.2026-08-24-phaseA-run3-convs4`):
  **9/16 corrupted.** `GPU had 0 tok` count = 13 (real restores). Pattern:
  c0 (always resident, never evicted — 4 conversations don't all fit, c0
  happened to stay in the working set) always clean (`'ready'`/`'ok'`,
  ~2s turns); c1/c2/c3 (evicted and restored every round from r1 on, ~6.5s
  turns) corrupted from r1 onward, every time.
- Run 4 (repeat, same server, `qwen.log.2026-08-24-phaseA-run4-convs4`):
  **12/16 corrupted.** Same pattern exactly (c0 always clean, c1/c2/c3
  corrupted r1–r3).

Sample corrupted reply (run 3, r1/c1):
```
'rocrollo郡phine奉enje晕谢�azziptic.pub长相GORücksุง dati t...'
```

**Decision: CORRUPT.** Two consecutive high-exposure runs (9/16, 12/16)
reproduce the corruption with `VLLM_OFFLOAD_EAGLE_FALLBACK=0` reverted (i.e.
pre-Phase-B connector behavior, the upstream eagle mark-everything fallback
active). **Phase B (the eagle-fallback change) is exonerated — this is a
latent day-one restore bug**, not something introduced by the
`VLLM_OFFLOAD_EAGLE_FALLBACK=0` change. Phase C runs with full scope (all four
KV groups, not just mamba).

Cleanup: `micke-start.sh` restored via `git checkout -- micke-start.sh`
(VLLM_OFFLOAD_EAGLE_FALLBACK=0 back in place). Server left running for Phase
B/C log archaeology; fs cache dir left as populated by Phase A (not the stale
pre-session one).

## Phase B — settle history from existing logs

No archived `qwen.log` survives from 2026-08-22 (the file is gitignored,
overwritten on every restart, and no one archived that day's session before
it rotated). The only surviving archived log is
`qwen.log.evidence-mamba-free-never-fired.log` (found in
`~/.local/share/Trash/files/`, not deleted — still readable), covering
**08-23 00:12:31–07:06:02**, entirely *before* the Phase-B change
(`VLLM_OFFLOAD_EAGLE_FALLBACK=0` went live 08-23 ~08:18). So this window is a
genuine pre-Phase-B sample, just not the 08-22 window the plan hoped for.

Contrary to the plan's Gap-1 assumption ("check whether responses are
captured too — if only requests..."), the request-logging patch **does**
capture the streamed response in a `=== RESPONSE (streaming) ===` section —
turn N's own reply is available directly, no need to hunt for it embedded in
turn N+1's message array.

Grepping that window for genuine full restores (`hit N offloaded tokens
after 0 GPU hit tokens`) finds exactly **one**:

```
DEBUG 08-23 00:14:37 [scheduler.py:892] Request chatcmpl-ba8c05cfc18ca8b4-93a7e1c7 hit 6656 offloaded tokens after 0 GPU hit tokens
```

`requests/2026-08-23T00-14-36Z_chatcmpl-ba8c05cfc18ca8b4_...log`'s `RESPONSE
(streaming)` section for that exact request:

```
<think>
The user is saying they want me to proceed with session `8bd9ac000c06`,
which was interrupted by a network error. I need to recall what that
session was about. Since I don't have the transcript of that session in
this webui session, I should use session_search to find it.

Let me try searching for that session.
</think>

I'll look up that session to see where it left off.

<tool_call>
<function=session_search>
<parameter=query>
8bd9ac000c06
</parameter>
</function>
</tool_call>
```

Coherent — sensible reasoning, correctly-formed tool call, on-topic. **This
one pre-Phase-B full restore was clean.**

**Verdict:** consistent with Phase A's "Corrupt" finding, but not a clean
cross-check either way — this is a single small restore (6,656 tokens = 8
blocks, vs. the ~57–59K-token/70-block restores in the Phase-A/doc repro) in
a workload (`simulate_hermes_traffic`-style concurrent agent chatter) that
essentially never generates real eviction pressure: one full restore in 7
hours of traffic, across hundreds of requests. It neither confirms nor
contradicts "restore was broken pre-Phase-B too" — it only shows small
restores *can* be clean, which was already known (Phase A's default
`--convs 2` run also had one small-ish clean restore). Does not overturn
Phase A's decisive high-exposure verdict. Worth carrying into Phase C: size
or chunk-count may gate the defect (see anomalies).

## Phase C — checksum triangulation

**Instrument:** `patches/restore-corruption-checksum-debug.patch` (temporary,
gated behind `VLLM_KV_RESTORE_CHECKSUM=1`, zero cost when unset). Three hook
points, all crc32-over-raw-bytes:

- **(a) STORE_DONE** — `v1/kv_offload/cpu/gpu_worker.py`,
  `SingleDirectionOffloadingHandler.get_finished()`: CPU row content once a
  GPU→CPU store's `end_event` confirms completion.
- **(b) LOAD_SUBMIT** — same file, `transfer_async()`, CPU→GPU direction: CPU
  row content at submission (before the async copy starts).
- **(c) RESTORE_READ** — `v1/worker/mamba_utils.py`, `preprocess_mamba()`:
  when `prev_state_idx` falls back to `(num_computed_tokens - 1) //
  block_size` (no local `mamba_state_idx` history — true for both an offload
  restore and an ordinary local-prefix-cache-hit new turn), hash the GPU
  state block at that index, for every Mamba layer, before this step's
  forward pass reads it.

Also added `LOAD_JOB_BLOCKS` (symmetric to the existing `STORE_JOB_BLOCKS`
debug line) in `scheduler.py`'s `update_state_after_alloc()`, so a load
job's `job_id` can be correlated back to its `req_id`.

**Two instrumentation bugs found and fixed before the data was trustworthy**
(both would have produced a spurious 100%-mismatch signal, i.e. looked like
proof of corruption that wasn't there — recorded so a future session doesn't
have to rediscover them):

1. `register_kv_caches()` (`distributed/kv_transfer/kv_connector/v1/
   offloading/worker.py`) offloads Mamba layers as **one combined page per
   layer** (conv-state + temporal-state concatenated), not one page per state
   type. The first cut of (c) hashed each state type separately —
   guaranteed never to match (a)/(b)'s per-layer combined-page hash,
   regardless of correctness. Fixed: concatenate all of a layer's
   `kv_caches` state tensors before hashing once.
2. The CPU tensor's row stride is sized to the **padded**
   `CanonicalKVCacheTensor.page_size_bytes`, but only the first
   `CanonicalKVCacheRef.page_size_bytes` (unpadded — the offload layer
   already tracks this distinction internally) is real data; the GPU
   tensor's own natural per-block stride has no such padding. Hashing the
   full CPU row against the unpadded GPU block gave a constant 8,192-byte
   discrepancy (`1,703,936` vs `1,695,744` bytes) and, again, a guaranteed
   mismatch. Fixed: truncate the CPU-side hash to `unpadded_bytes`.

**Clean single-process run** (no restart in between, FS tier verified empty
at start, so store→load→read all trace through one self-consistent
allocation lifetime): `test_kv_cache_identity_corruption.py --convs 3
--rounds 2`. Round 0: 3 fresh ~58K-token conversations, genuine ~61s cold
computes (confirms FS tier really was empty). Round 1: `c0` full restore
(`hit 58240 offloaded tokens after 0 GPU hit tokens`) → **clean** (`'ok'`);
`c1` full restore, same size → **corrupted**; `c2` mostly-resident (54,080
GPU + 4,160 loaded) → clean. Deterministic — repeated once, identical
outcome and identical corrupted text both times.

**Result — every single (group, layer) row matched, for both requests,
across the entire pipeline:**

```
c0 [CLEAN]     group=0 layer=0 gpu_blk=178: LOAD-vs-READ(b-vs-c)=MATCH | tidx=0 cpu_blk=276: STORE-vs-LOAD=MATCH ... (×16 tidx)
c1 [CORRUPTED] group=0 layer=0 gpu_blk=...: LOAD-vs-READ(b-vs-c)=MATCH | ... STORE-vs-LOAD=MATCH (×16 tidx)
```
144/144 (c) rows found a matching (b) crc (all 3 Mamba groups × ~16 layers ×
3 requests); all their (b) crcs in turn found a matching (a) crc among that
`(cpu_blk, tidx)`'s store history. **(a)=(b) held for all 13,650 store/load
rows across all four KV-cache groups** (mamba AND attention — gpu_worker.py's
instrument is group-agnostic; 70 chunks × 65 total layers × 3 convs =
13,650, confirming full 4-group coverage on the store/load side specifically).

Reading Phase C's classification table with this data: row 1 ("(a)≠(b),
RAM-tier corruption") is **ruled out for all four groups**. Rows 2/3
("(b)≠(c), load/placement wrong" / "(a)=(b), (c)≠(b), store captured wrong
content") are **ruled out for the three Mamba groups specifically** — the
exact bytes stored are the exact bytes a corrupted request's own first
forward step reads back. That leaves row 4, "all equal; corruption is
elsewhere" — but the "elsewhere" reference comparison (recompute the same
prefix with the connector fully disabled, same boundary, compare) could not
be completed as designed: see below.

**Reference-recompute attempt (partial, inconclusive on its own terms, but
surfaced a real anomaly):** restarted with `--kv-transfer-config` dropped
entirely (checksums still on) and replayed conversation `c1`'s exact
prompt content (`build_prompt(1)`, byte-identical) as a lone, uncontended
two-turn conversation. Reply was **coherent** (`'ok'`) — consistent with the
doc's established "connector disabled = always clean" baseline. But its own
local prefix-cache hit landed at **57,408 tokens (block 69)**, not the
**58,240 tokens (block 70)** the offload path both stores and restores from
— one full 832-token chunk short. So this run never actually exercised block
70 as a cold-resume boundary, and the checksums it produced (`computed=57408
prev_state_idx=68`) aren't at the boundary that matters; no byte-for-byte
reference value for block 70 was obtained.

That one-block gap is itself worth chasing (see Anomalies): it means
ordinary local prefix-caching does **not** trust the newest complete
Mamba/GDN chunk for reuse, while the offload connector does. Read
`docs/mamba-align-prefill-leak.md`'s `mamba-align-stale-checkpoint-offload-
slack.patch` section — `MambaManager.remove_skipped_blocks()` frees a
superseded checkpoint within 1-2 of the *owning request's own* scheduling
steps, independent of any offload store. That patch exists to give the
connector a slack window to back up a checkpoint before this same-request
freeing reclaims it, and was left unwired because it didn't reliably reduce
HIT_DIVERGED counts under concurrent load. **It was never tested against
output coherence for a deterministic single-boundary restore, which is a
more direct test of exactly this race** — so it was tried here:

**`VLLM_MAMBA_ALIGN_STALE_SLACK_BLOCKS=2` test (negative result):**
`test_kv_cache_identity_corruption.py --convs 4 --rounds 3`, checksums still
on, fresh FS tier. **6/12 corrupted** — same rate and the same c0-clean /
c1,c2,c3-corrupted pattern as every other high-exposure run in this
investigation (Phase A's runs, both non-slack). The slack window does not
change corruption at all, in either direction. This doesn't fully rule out
a same-request-freeing race (slack=2 might be insufficient, or the real
window closes faster than any slack budget can cover), but it removes the
leading hypothesis from "confirmed" back to "unconfirmed," and — combined
with (a)=(b)=(c) holding even for corrupted restores — argues against "the
Mamba checkpoint block gets reused/overwritten before or during its store"
as the mechanism, since that would be expected to show up as a checksum
mismatch, which it didn't.

## Phase D — spec-boundary optimistic-store check

*Not run.* Given Phase C didn't establish a root cause to build on, and §3's
known defect is explicitly scoped to decode-boundary crossings (this
repro's boundary is crossed during prefill, which the plan itself already
flags as not explaining the current symptom), implementing and testing the
`STORE_AHEAD_OF_VERIFY` logging was deprioritized in favor of exhausting the
Phase C leads first. Worth doing in a follow-up session regardless, as its
own independent decode-path bug per the plan's §3.

## Fix

*Not implemented — root cause not established.* No workaround-as-fix
(disabling offloading) is proposed, per the plan's ground rules. The
`VLLM_MAMBA_ALIGN_STALE_SLACK_BLOCKS` knob was tested as a candidate fix and
found ineffective (see Phase C); it is not being wired into `micke-start.sh`.
`VLLM_OFFLOAD_EAGLE_FALLBACK` stays unchanged (`=0`, default/production,
matches pre-investigation state) — Phase A showed it has no bearing on this
bug.

## Anomalies & open questions

**Full elimination table** (this session, all with evidence above):

| candidate cause | status | evidence |
|---|---|---|
| `VLLM_OFFLOAD_EAGLE_FALLBACK` change (Phase B) | **ELIMINATED** | Phase A: corrupts identically with the fallback reverted to pre-Phase-B behavior |
| RAM-tier byte corruption in residency, any of 4 groups | **ELIMINATED** | Phase C: (a)=(b) for all 13,650 store/load rows, mamba + attention |
| Load transfer / GPU placement wrong, Mamba groups 0-2 | **ELIMINATED** | Phase C: (b)=(c) for all 144 read rows, clean and corrupted alike |
| Store captured wrong content, Mamba groups 0-2 | **ELIMINATED** | same — (a)=(b)=(c), so whatever's in the store is faithfully what's read back |
| Mamba checkpoint freed by same-request progress before store completes | **UNCONFIRMED, LEANS AGAINST** | slack=2 test: no effect on corruption rate |
| §3 optimistic storability across spec-decode boundaries | **OUT OF SCOPE for this repro** | plan already notes this repro's boundary is a prefill crossing, not decode; Phase D not run |
| Attention group (index 3) restore content | **NOT TESTED** | no GPU-read-side check exists for attention (no "cold resume" hook was added there) |
| Original hypotheses ruled out pre-session (threshold, vision, MAMBA_FREE, O_DIRECT, disk cap, NVMe tier, mixed cache) | still eliminated | see `docs/kv-offload-cache-identity-corruption.md`, not re-chased |

**Sharpest remaining hypothesis: the bug is not in the Mamba/GDN state at
all.** Every Mamba-side check came back clean, including for the corrupted
request. Three groups are Mamba/GDN; the fourth is full attention, and it
was checksummed store→load (a=b, clean) but **never** checked load→GPU-read,
because attention doesn't go through `preprocess_mamba`'s cold-resume path —
no equivalent hook was written for it this session. **Deciding experiment:**
find or add attention's own resume/placement code path (likely in the
general `KVCacheManager`/attention-group restore logic, not
`mamba_utils.py`) and add the same GPU-side checksum there, keyed the same
way, then re-run the exact clean-run recipe above (`--convs 3 --rounds 2`,
fresh FS tier, one uninterrupted process) and check whether attention's
restored KV, not Mamba's, is where (b)≠(c) finally shows up.

**Second lead, not yet chased: the one-chunk gap between local and offloaded
resume boundaries.** For identical content, local prefix-cache reuse stops
at 57,408 tokens (block 69) while offload restore uses 58,240 (block 70) —
offload is willing to serve a chunk that ordinary local reuse doesn't trust.
Whether that's expected (offload's own storable-chunk accounting is
legitimately allowed to be one ahead of local block-hash exposure) or itself
a bug wasn't determined. **Deciding experiment:** trace exactly when block
70 gets hash-registered (or not) for local reuse vs when `_build_store_jobs`
marks it storable — read `single_type_kv_cache_manager.py`'s `MambaManager`
alongside `_calc_num_offloadable_tokens`/`storable_chunks` in
`offloading/scheduler.py`, side by side, for this exact boundary.

**Third lead: get a real reference value at block 70.** The reference-recompute
attempt above landed one block short by accident (local resume boundary,
not the connector-disabled equivalent of block 70). To get the boundary
that actually matters, either (a) force the connector-disabled run's turn 2
prompt to be *shorter* so its own local hit is forced to land exactly at
58,240 (e.g. truncate the injected reply so the shared prefix is exactly
58,672 tokens and no more), or (b) add a one-off checksum print keyed
directly off `num_computed_tokens` at the point turn 1's own prefill first
crosses 58,240, with the connector disabled, rather than relying on a later
turn's cold-resume fallback to land there naturally.

**Housekeeping for a follow-up session:** the checksum debug patch
(`patches/restore-corruption-checksum-debug.patch`) is still applied to the
live venv and kept in `patches/`, gated behind `VLLM_KV_RESTORE_CHECKSUM=1`
(default off, zero cost) — safe to leave applied; `bash verify.sh --no-server`
passes with it in place. The server was left running in default production
config (`bash micke-start.sh`, no debug env vars) at the end of this
session. Several `qwen.log.*` snapshots from this investigation are on disk
in the repo root (not committed — large, and `.gitignore` already excludes
`qwen.log` itself); the ones worth keeping for a follow-up are
`qwen.log.phaseC-clean-run` (the definitive triangulation run) and
`qwen.log.phaseC-reference-run` (the boundary-mismatch discovery) — the
`*-attempt1/2/3-*` and `*-slacktest-*` files are superseded/negative-result
evidence, safe to delete once this doc's excerpts are trusted.

- The GPU KV cache pool size (171,956 tokens) does not match the ~6.11 GiB
  figure implied by the original doc's repro description ("only about two
  fit"). Worth reconciling — possibly a different `CTX`/dtype config at the
  time the doc was written, or the doc's figure was approximate. Not chased
  further since `--convs 4` gives unambiguous exposure regardless.
- c0 was clean in every single run across all four Phase-A runs, including
  the two 4-conv corrupting runs, despite being restored in run 1 (r2/c0,
  clean) — so residency status alone doesn't fully explain it; timing /
  eviction-order effects (c0 is always the first conversation, first primed,
  possibly always the one that stays in the working set under the eviction
  policy) need to be kept in mind for Phase C's checksum design (don't assume
  "conversation index" correlates with anything causal — it's "which one gets
  evicted" that matters, and that happened to always be c1/c2/c3 here).
