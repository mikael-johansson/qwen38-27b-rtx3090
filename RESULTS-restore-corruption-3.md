# Results 3: the uninitialized-memory (stale-block) hypothesis

Executed per `RESTORE_CORRUPTION_PLAN3.md`. Branch `restore-corruption`,
continuing from `RESULTS-restore-corruption-2.md`.

## TL;DR

**Hypothesis CONFIRMED, with real exposure, using two independent scribble
patterns.** A checkpoint page contains byte range(s) the live compute path
never writes but the restore path consumes. On a virgin GPU allocation
those bytes are CUDA-zeroed (harmless); when a physical block was
previously used by another conversation and gets recycled, they hold that
conversation's stale bytes (harmful). Deliberately filling every KV-cache
tensor with nonzero garbage right after allocation turns a previously
**always-clean** isolated-priming-then-restore (plan 2's A5, 3/3 clean)
into a **corrupted** one — with a mild pattern the effect was numerically
inert (still clean), but with two different "loud" patterns (0xFF/NaN and
0x7B/finite-huge) it reliably corrupted, with genuine full-restore
exposure confirmed every time.

**Field bisection (Phase B) is inconclusive — not because the technique
failed, but because it can't distinguish the fields from output text
alone.** Scribbling *only* the Mamba/GDN groups reproduces corruption;
scribbling *only* the attention group, independently, reproduces
**byte-for-byte identical** corrupted output — with both the NaN pattern
and the finite pattern. This means at least one of {Mamba, attention} has
a genuine unwritten-but-consumed region; it does not prove which, or rule
out both. The identical text across different corruption sources looks
like generic autoregressive collapse (repetitive-token loops are a common
LLM failure mode once hidden state is far enough out-of-distribution), not
evidence the two fields are somehow the same bug.

**Phase C (mechanism) — the plan's own leading candidate is ruled out; the
real gap is not yet found.** The plan speculated the culprit was the
Mamba conv-state sliding-window copy (`state[dst, :conv_width -
token_bias]`) leaving a `token_bias`-sized remainder unwritten. Traced
directly: for a plain-prefill boundary crossing (this repro's case —
already established in session 1 that the corrupting boundary is crossed
during prefill, not decode), `accept_token_bias` is **always** computed
as `int(input_batch.num_accepted_tokens_cpu[i]) - 1`, and
`num_accepted_tokens_cpu[i]` is unconditionally reset to `1` after every
boundary crossing (`mamba_utils.py:1209`) — so for a pure-prefill request
that never goes through spec-decode acceptance, `accept_token_bias == 0`
at every crossing, which means the copy writes the *full* `conv_width`
every time. No remainder gap in this path, for this repro. The temporal/
SSM-state copy also always writes its full `state_inner_size` regardless
of `token_bias`. **The exact write-path gap that the scribble test proves
exists has not been located.** See Anomalies for the concrete next
experiment (read back live-computed bytes directly, rather than inferring
the gap from corrupted-output text).

No fix implemented — mechanism not yet localized precisely enough to fix
narrowly.

## Phase A — the scribble test

**Infrastructure note (real bug, unrelated to the hypothesis, cost real
time this session):** partway through, isolated-priming replays started
silently falling back to full recompute instead of restoring — the
offload lookup got stuck in a permanent `RETRY`/`any_uncertain=True`
state (`_sliding_window_lookup`/`_maximal_prefix_lookup` in
`offloading/scheduler.py`) that never resolved, regardless of how long the
test waited after the process came up healthy. Traced to: `/d/nvme_cache`
was **100% full** (14MB free of 234GB) from three sessions' worth of
`mv`-ed-aside cache backups (`vllm_kv_stashed/`, 222GB) that were never
cleaned up. Confirmed by reproducing the stuck-RETRY symptom with a
completely vanilla (unscribbled) recipe once the disk was full, and
confirming it disappeared immediately after clearing the stashed
directories (user-approved `rm -rf`). Not a KV-offload correctness bug —
a disk-space operational issue this investigation's own methodology
(repeatedly stashing multi-GB caches instead of deleting them, per the
plan's ground rules) caused. Housekeeping: don't let `vllm_kv_stashed`
accumulate indefinitely in a future session; the plan's "move aside, don't
delete" rule protects the *live* cache from an in-progress test, not
backups from tests that already finished and whose evidence has been
extracted into logs.

**Hook:** `patches/restore-corruption-scribble-debug.patch`,
`VLLM_DEBUG_SCRIBBLE_KV={1|ff|big}` — fills every KV-cache tensor with a
nonzero byte pattern in `GPUModelRunner.initialize_kv_cache()`, right
after allocation and before the offload connector ever registers the
tensors. `VLLM_DEBUG_SCRIBBLE_KV_ONLY={attn|mamba}` restricts which
KV-cache groups get scribbled, for Phase B.

**A2 (unscribbled control, this session, clean disk):** isolated
`c1` priming, restart, replay — clean, with confirmed full-restore
exposure:
```
turn1: prompt=58672 cached=0  7.4s 'ready'
turn2: prompt=58696 cached=0  1.1s 'ok'
KV LOAD ... | fs -> RAM | 280 blk; hit 58240 offloaded tokens after 0 GPU hit tokens; RAM -> VRAM | 70 blk / 58240 tok (GPU had 0 tok)
```
Matches plan 2's A5 exactly — environment had not drifted.

**A1 (`VLLM_DEBUG_SCRIBBLE_KV=1`, pattern 0x3C, fp16 ≈1.06):** live
priming reply correct (`'ready'`, 60.7s — the live path is immune, as
predicted, since it only reads what it itself wrote). Restart, replay:
```
turn1: prompt=58672 cached=0  7.4s 'ready'
turn2: prompt=58696 cached=0  1.1s 'ok'
```
Clean, with confirmed exposure (`GPU had 0 tok`, 70 blk / 58240 tok both
turns). The mild pattern turned out to be numerically inert here.

**A3 (`VLLM_DEBUG_SCRIBBLE_KV=ff`, pattern 0xFF, NaN in fp16/fp8):** live
reply still correct (`'ready'`, 61.2s). Restart, replay:
```
turn1: prompt=58672 cached=0 13.5s 'AKElothAKE光华filer自闭恭ariahickerascriptхинаInitStructanjiananjianlifyilienanjianPP'   <<< CORRUPTED
turn2: prompt=58696 cached=0  5.3s '/bowereldainnamon<|box_end|>ety[][小妹aintyPPERwthivable...'                          <<< CORRUPTED
```
Confirmed exposure: `KV LOAD ... fs -> RAM | 280 blk`, `hit 58240
offloaded tokens after 0 GPU hit tokens`, `RAM -> VRAM | 70 blk / 58240
tok (GPU had 0 tok)`, both turns. **The exact same recipe that was clean
3/3 in plan 2's A5, with the only change being the pool's unwritten bytes,
now corrupts.** This is the decisive result: hypothesis confirmed.

## Phase B — field bisection

All four runs below: isolated `c1` priming with the selector active,
restart, replay, confirmed `GPU had 0 tok` / `70 blk / 58240 tok`
exposure every time.

| pattern | selector | live priming reply | restored reply |
|---|---|---|---|
| 0xFF (NaN) | `mamba` | `'ready'` (60.7s) | **CORRUPTED** — `'AKElothAKE光华filer自闭恭aria...'` (byte-identical to the all-groups 0xFF run) |
| 0xFF (NaN) | `attn` | `'ready'` (60.8s) | **CORRUPTED** — same text, byte-identical again |
| 0x7B (finite, fp16 61280.0) | `mamba` | `'ready'` (60.8s) | **CORRUPTED** — `'大局消ysysres Lifres lif...'` (a *different* corrupted string — repetitive-token pattern, not the NaN-run's text) |
| 0x7B (finite, fp16 61280.0) | `attn` | `'ready'` (61.3s) | **CORRUPTED** — byte-identical to the mamba/0x7B run |

Introduced the 0x7B pattern specifically to rule out "NaN just propagates
identically through the residual stream regardless of source, so of course
the two selectors look the same" — a finite (non-NaN) pattern should not
have that universal-contamination property. It corrupted identically
anyway (mamba vs. attn, same text, both 0x7B runs). This looks like
generic out-of-distribution collapse into a repetition loop (`Lif`,
`res`, `orama` recurring) rather than a shared corruption mechanism —
a known LLM failure mode once hidden state is sufficiently perturbed,
independent of which component introduced the perturbation.

**Conclusion: at least one of {Mamba/GDN groups, attention group} has a
genuine unwritten-but-consumed region on restore. This test cannot
determine whether it's one, the other, or both** — text-level comparison
of already-collapsed output carries no positional information. A
different technique is needed (see Anomalies).

## Phase C — mechanism

Traced the plan's specific candidate (`_copy_mamba_state_block`'s
conv-state slide, `mamba_utils.py`) for the cold-resume / boundary-
crossing-during-prefill case, which is what this repro's boundary
(58,240 tokens, block index 69) actually is (established session 1):

- `preprocess_mamba` (`mamba_utils.py:~1180-1210`): on a boundary
  crossing, `accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i])
  - 1`, then unconditionally resets `input_batch.num_accepted_tokens_cpu[i]
  = 1` (line 1209) regardless of the fused/non-fused path.
- For a request that is purely prefilling (no spec-decode
  accept/reject ever runs on it — prefill doesn't draft), nothing else
  ever changes `num_accepted_tokens_cpu` away from `1` between crossings,
  so `accept_token_bias == 0` at every single prefill-time boundary
  crossing, including the one that creates the offloaded chunk at
  58,240.
- With `token_bias == 0`: the conv-state copy's destination range is
  `state[dst, :conv_width - 0]` = the *entire* conv window — fully
  written, no remainder. The temporal/SSM-state copy
  (`_copy_mamba_state_block`'s non-conv branch) always copies the full
  `state_inner_size * state_elem_size` regardless of `token_bias` — also
  always fully written.

**This specific mechanism does not apply to this repro.** It would
matter for a boundary crossed *during decode with active spec-decode
acceptance* (a genuinely different, real scenario — not what's being
chased here), but not for a plain-prefill crossing. The scribble test's
result stands regardless (uninitialized memory getting read is proven by
direct construction, independent of this specific narrower hypothesis
about *why*) — the actual gap is elsewhere and was not located this
session.

## Fix

Not implemented. Localizing the exact unwritten byte range (which field,
which sub-region, which write path skips it) is a prerequisite for a
narrow fix per the plan's Phase D ordering (fix the write, not just zero
things defensively at capture time), and that localization is the
unfinished work — see Anomalies.

## Anomalies & open questions

**Disk-full stuck-lookup bug** (see Phase A) — not a KV-offload
correctness issue, but worth a permanent fix regardless: a full
secondary-tier disk should produce a clean MISS or an error, not a
lookup that hangs in `RETRY` forever and silently masks itself as "just
recompute, nothing's wrong." If this investigation's own `vllm_kv_stashed`
habit could trigger it, real production disk pressure could too. Worth a
follow-up issue independent of the corruption investigation.

**Sharpest remaining hypothesis and the deciding experiment:** the
scribble test proves the phenomenon exists but corrupted-text comparison
can't localize *which* field or *which byte range*. The direct fix:
instead of inferring the gap from downstream text, **read back the raw
bytes**. Recipe for a follow-up session:
1. Scribble (pick one field at a time, `big` pattern for a distinctive,
   non-zero, non-NaN signature) an isolated `c1` priming as in Phase B.
2. Before restarting — while the *same* live process still holds the
   just-computed checkpoint block — dump the raw bytes of that block
   (extend the existing `RESTORE_CHECKSUM`-style hooks, or a one-off
   debug read, in `v1/kv_offload/cpu/gpu_worker.py`'s store path, right
   after the GPU→CPU copy: instead of hashing, count/report *how many
   bytes still equal the scribble pattern* and at what byte offsets/
   ranges within the page).
3. Any surviving `0x7B7B...` run pinpoints the exact unwritten range —
   which sub-tensor (conv vs. temporal for Mamba; K vs. V, or a specific
   layer, for attention), and its offset/length. That directly answers
   "who never writes this" without needing corrupted-output inference at
   all, and sidesteps the field-bisection ambiguity entirely since the
   byte range itself will say which field it's in.
4. Once the range is known, find its writer(s) by symbol/grep (the
   plan's original Phase C item 1) and follow the plan's Phase D fix
   ordering.

**Note for whoever picks this up:** don't re-chase the conv-`token_bias`
remainder hypothesis for prefill-boundary crossings — traced and ruled
out this session (Phase C above). It may still be real for a
decode-time, spec-decode-active boundary (a different scenario, not this
repro) and is worth keeping in mind for that case specifically, per plan
1 §3's separate optimistic-storability finding, which remains open and
unrelated to this investigation's main repro.

**Housekeeping:** server restored to clean default production config, no
debug env vars, at end of session. Both debug patches
(`restore-corruption-checksum-debug.patch`,
`restore-corruption-scribble-debug.patch`) remain applied to the live
venv — both are off by default and `verify.sh --no-server` passes with
them in place. `/d/nvme_cache/vllm_kv_stashed` was fully cleared this
session (user-approved) and should be watched going forward — a follow-up
session doing more isolated-priming cycles will regenerate stash
directories quickly; clear them proactively rather than letting them
reach 100% disk again. `qwen.log.*` snapshots from this session are on
disk in the repo root (not committed, large); the load-bearing ones are
prefixed `plan3-A1-`, `plan3-A2-`, `plan3-A3-`, and `plan3-B-`.
