# Plan 3: restore corruption — the uninitialized-memory (stale-block) hypothesis

> **For the executing agent (Opus/Sonnet):** self-contained follow-up to
> `RESTORE_CORRUPTION_PLAN.md` (its §4 ground rules apply unchanged),
> `RESTORE_CORRUPTION_PLAN2.md`, and `RESULTS-restore-corruption-2.md`.
> Write findings to `RESULTS-restore-corruption-3.md`. Branch:
> `restore-corruption`.

## 0. Reframing plan 2's conclusion (important — read carefully)

Plan 2's results concluded "the forward computation itself diverges under
contention." That framing is contradicted by the session's own data: in
the contended baseline, **c1's live turn-1 reply was correct** (`'ready'`)
— the forward computation was right. Only the *stored checkpoint*,
restored later, produces soup. The correct statement is:

> The captured checkpoint page contains region(s) the **live** compute
> path never reads but the **restore** path does. Those regions are
> whatever happened to be in the physical block beforehand: zeros on a
> virgin pool (freshly mapped GPU pages — driver-zeroed), a dead
> conversation's stale state on recycled blocks.

This is the classic uninitialized-memory signature, and it explains every
observation at once:
- c0 always clean (always primed on a virgin pool; unwritten regions = 0).
- c1/c2/c3 always corrupt when primed after another conversation
  (unwritten regions = predecessor's stale bytes).
- A5 isolation sufficient (all primings virgin).
- Live-correct / restore-corrupt (live path keeps its own running state;
  restore path consumes the full stored page).
- Byte-identical corrupted text across runs and sessions (the stale
  content is the *deterministic* previous conversation's state —
  deterministic garbage in, deterministic soup out; token-level
  determinism survives the bit-level CRC jitter plan 2 documented).
- Chunk schedule identical isolated-vs-contended (scheduling was never
  the variable; block *contents at allocation* were).
- Retention slack no effect; eagle fallback no effect (neither touches
  allocation-time block contents).

Where the unwritten region plausibly lives (background for Phase C, do
not assume — measure first): the mamba align machinery does
**partial-region block writes** — `_copy_mamba_state_block` /
`postprocess_mamba_fused_kernel` in `$SP/v1/worker/mamba_utils.py` copy
the conv state as a slid window (`state[src, token_bias:]` →
`state[dst, :conv_width - token_bias]`, leaving the remainder of the
destination rows unwritten) and copy temporal state as a separate region;
`new_block_ids_to_zero` exists precisely because *some* blocks need
zero-init, so its coverage having a hole is a natural suspect. The stored
chunk is the full unpadded page (all of a layer's state tensors
concatenated), so any never-written byte in the page travels to the tier
and comes back on restore.

## 1. Phase A — the scribble test (decisive; one debug hook + two runs)

**Hook (temporary patch, `patches/restore-corruption-scribble-debug.patch`):**
after the GPU KV cache tensors are allocated and registered (post
`initialize_kv_cache` in the worker — after the cumem allocator has done
its thing, operating on the live tensors), if env
`VLLM_DEBUG_SCRIBBLE_KV` is set, fill **every** KV cache tensor with a
nonzero byte pattern and log one checksum line per tensor proving it
happened. Pattern: byte `0x3C` (fp16 `0x3C3C` ≈ 1.06, fp8 ≈ mild values
— deliberately *not* NaN, to mimic plausible stale data rather than
poison that might crash paths stale data wouldn't). Keep a second
pattern, `VLLM_DEBUG_SCRIBBLE_KV=ff` (NaN-heavy), in reserve for Phase A3.

**A1 — scribbled isolated priming:**
1. Fresh fs tier dir. Start server with `VLLM_DEBUG_SCRIBBLE_KV=1`.
2. Prime **only c1** (`build_prompt(1)` turn 1), exactly like plan 2's A5
   isolated recipe. Record the live reply (prediction: still `'ready'` —
   the live path should be immune; if the live reply is ALREADY corrupt,
   that's a different, even more interesting result — record and
   continue).
3. Restart **without** scribble (plain config). Replay c1's turn(s);
   confirm full-restore exposure (`GPU had 0 tok`).
4. Read the reply.

**Corrupt → hypothesis confirmed**: the only change from plan 2's A5
(clean 3/3) is that the accidental zeros under the priming were replaced
with garbage. **Clean → hypothesis wrong** — go to Phase E (fallbacks)
instead of Phase B/C/D.

**A2 — unscribbled same-session control:** repeat steps 1–4 without the
env var (this is plan 2's A5, re-run once in this session for a clean
side-by-side). Must be clean; if not, something else drifted — stop and
report.

**A3 (only if A1 was unexpectedly clean):** retry with the `ff` pattern
before declaring the hypothesis dead — a mild pattern may be numerically
benign where real stale state isn't.

## 2. Phase B — bisect the field (2–4 runs, selective scribble)

Extend the hook: `VLLM_DEBUG_SCRIBBLE_KV_ONLY=<selector>` scribbles only
a subset, everything else untouched:
- `mamba_conv` — only conv-state tensors of the mamba groups,
- `mamba_temporal` — only temporal/SSM-state tensors,
- `attn` — only the attention group's pages.

(Identify the tensors via the same per-layer state metadata the copy
funcs use — `get_mamba_state_copy_func` / the kv_caches dict; log what
was scribbled.) Re-run the A1 recipe per selector. The selector(s) that
reproduce corruption localize the field. If `attn` reproduces, the
attention path has its own uninitialized-region consumer — follow it
instead of/in addition to mamba.

## 3. Phase C — read the code, name the mechanism

For the guilty field, produce the write/read narrative with file/symbol
references:
1. Who writes that region during priming (kernel or copy func), and
   exactly which sub-range is left unwritten (the conv `token_bias`
   slide remainder, an alignment pad, a whole-tensor never-write on some
   path, etc.).
2. Who reads it on restore (the `preprocess_mamba` resume copy, the
   first forward's state load, attention metadata) and why the live path
   doesn't.
3. Why `new_block_ids_to_zero` (or whatever zero-init machinery exists —
   find its writers and its trigger conditions) does not cover this
   allocation route. c0-on-virgin-pool works by *accident*, so the
   zero-init either doesn't exist for this case or doesn't fire.

## 4. Phase D — fix + gates

Fix at the narrowest correct point — prefer, in order: (a) make the
partial-write complete (zero-fill the destination remainder in the copy
kernel/func at boundary-checkpoint creation), (b) zero the relevant
region at block allocation for the paths that consume it (extend the
existing zero-init machinery), (c) zero at store-capture time (mask the
store) — (c) only if (a)/(b) are shown impractical, since it leaves the
GPU-side inconsistency in place. Document why the chosen point is
sufficient (the Phase C narrative must say who else could read the
region).

Gates (all of):
1. The A1 scribble recipe itself now comes back **clean** — this is the
   new strongest gate, it removes the virgin-pool luck entirely.
2. `test_kv_cache_identity_corruption.py --convs 3 --rounds 2` and
   `--convs 4 --rounds 3`: 3 consecutive clean runs with exposure proof.
3. Plan 1 §9's remaining gates: coherence-checking regression script,
   eviction-variant needle test, 15-min 3-way simulate run,
   `verify.sh --no-server` green.
4. Scribble/checksum debug patches reverted from the venv (files kept in
   `patches/`, headers marked temporary-diagnostic).
5. Docs updated: `docs/kv-offload-cache-identity-corruption.md` →
   FIXED with the mechanism; correct plan 2's results framing ("forward
   computation diverges" → the checkpoint-region statement above) in an
   addendum; note the cross-run-CRC-invalidity lesson stays valid.

## 5. Phase E — fallbacks (only if Phase A refutes the hypothesis)

1. Plan 2's never-run toggles, in order: no-async-scheduling, no
   spec-decode, `MAX_SEQS=1` (each one contended baseline run).
2. The same-run drift detector (plan 2 Phase C item 2): crc GPU src at
   store-job creation vs at copy completion, contended run, c1's
   boundary chunk. Same-run comparisons only (cross-run CRC is invalid,
   per plan 2's control).
3. Report per plan 2's rules: updated elimination table + the single
   sharpest next experiment. Do not spiral into new instrumentation
   beyond these without reporting first.

## 6. Results file format (`RESULTS-restore-corruption-3.md`)

- **TL;DR**: A1 verdict (hypothesis confirmed/refuted), guilty field if
  bisected, mechanism if named, fix + gate status.
- **Phase A**: A1/A2(/A3) outcomes with the scribble-proof checksum
  lines, exposure lines, and verbatim replies (live priming reply AND
  restored reply — the live one matters, see A1 step 2).
- **Phase B**: selector → outcome table.
- **Phase C**: the mechanism narrative with file/symbol references and
  the exact unwritten byte-range.
- **Phase D**: the fix diff summary + every gate's result.
- **Anomalies/open questions**: raw evidence over interpretation, as
  before. Note anything that suggests other consumers of the same
  uninitialized region (potential sibling bugs — e.g. whether *local*
  reuse paths could ever read such a region, and why they demonstrably
  haven't).
