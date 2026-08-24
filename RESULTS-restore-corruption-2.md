# Results 2: restore corruption — store-side vs restore-side split

Executed per `RESTORE_CORRUPTION_PLAN2.md`. Branch `restore-corruption`,
continuing from `RESULTS-restore-corruption.md`.

## TL;DR

**H-STORE confirmed, with a correction to the plan's own framing of it.**
The corrupted bytes are already wrong at the moment they're captured for
storage — replaying already-stored bytes on a totally fresh, idle,
zero-contention server still corrupts (2/2). But the *mechanism* is not an
eviction/reuse race clobbering otherwise-correct bytes: priming the exact
same conversation in complete process isolation (no other conversation ever
sharing memory with it) produces bytes that restore **clean**, every time
(3/3 conversations). Since a=b=c already held (session 1) even for a
corrupted restore — meaning nothing mutates the bytes between store
completion and GPU-read — the two results together mean **the forward
computation itself, at the moment `c1`'s boundary chunk is computed, produces
a different (and sometimes wrong) result depending on whether other
conversations have shared the process's memory pool before it** — not that
correct bytes get physically corrupted afterward. Root cause still not fully
localized (which mechanism produces the wrong computation), but the search
space is now much smaller: it's during priming, it's contention-dependent,
and it is not a RAM-tier data-movement problem. See Anomalies for the
sharpest next experiment. No fix implemented.

## Phase A

### A1 — the master fork (restart-replay)

**Setup:** fresh FS tier, default production config. Baseline run
(`test_kv_cache_identity_corruption.py --convs 3 --rounds 2`): usual outcome
— `c0`/`c2` clean, `c1` corrupted, byte-identical corrupted text to every
prior run in this investigation (including a run from a completely separate
session hours earlier). Archived (`qwen.log.2026-08-24-plan2-A1-baseline-full-run2`
and `-run2`).

**Replay:** killed the server, restarted (RAM tier gone, FS tier persists —
`PYTHONHASHSEED=0` makes the stored chunks reachable again), confirmed idle
(`curl /health`, no other requests), then replayed **only `c1`'s two turns**
— byte-identical content (`build_prompt(1)`, exact turn-1 reply `'ready'`
taken from the baseline run's own request log) — against the fresh, solo
process.

```
turn1: prompt=58672 cached=0 13.2s '联ditigar-addon령ester上空克拉求精vik太和فسviderurdy Singertipsmas.getTotalpre…'   <<< CORRUPTED (turn 1 itself!)
turn2: prompt=58696 cached=0  5.3s '� FultonGenerationStrategyessin Right渣og Giri索性p*@只有自己跻arez苍icatomp…'      <<< CORRUPTED
```

Exposure confirmed genuine both turns:
```
KV LOAD  chatcmpl-a2145d2587ce881d-8754b839 | fs   -> RAM  | 280 blk
Request chatcmpl-a2145d2587ce881d-8754b839 hit 58240 offloaded tokens after 0 GPU hit tokens
KV LOAD  chatcmpl-a2145d2587ce881d-8754b839 | RAM  -> VRAM |  70 blk /  58240 tok (GPU had 0 tok)
KV LOAD  chatcmpl-a244738401fec3b9-ad59e481 | fs   -> RAM  | 280 blk
Request chatcmpl-a244738401fec3b9-ad59e481 hit 58240 offloaded tokens after 0 GPU hit tokens
KV LOAD  chatcmpl-a244738401fec3b9-ad59e481 | RAM  -> VRAM |  70 blk /  58240 tok (GPU had 0 tok)
```

Turn 1's own reply — which the baseline run computed *fresh* and got right
(`'ready'`) — comes back corrupted when the exact same content is served via
a pure FS-tier restore on a brand-new process. **Corrupt on a fresh, idle
server → H-STORE.**

**Repeated for confidence: identical result, byte-for-byte identical
corrupted text, both turns, second independent baseline+replay cycle.**
2/2. (`qwen.log.2026-08-24-plan2-A1-replay1-run`,
`qwen.log.2026-08-24-plan2-A1-both-confirmed`.)

A2–A4 (mechanism-class toggles: async-scheduling, spec-decode, MAX_SEQS=1)
were **not run** — A1 and A5 (below) already established the key variable
(store-time contention vs. isolation) more directly and with a cleaner
signal than any of the three toggles would have added on their own; time
was spent on A5 and the CRC drill-down instead. Worth doing in a follow-up
if the sharpest hypothesis (Anomalies) doesn't pan out.

### A5 — serialized-priming test (run early, out of order, because it's the
cheapest way to settle whether contention at store time even matters)

Three separate server restarts, each priming **exactly one** conversation's
turn 1 in total isolation (empty pool, no other conversation ever loaded,
zero contention of any kind) against a freshly cleared FS tier that
*accumulates* across the three restarts:

```
prime c0: prompt=58669 cached=0 60.7s 'ready'
[restart]
prime c1: prompt=58672 cached=0 60.9s 'ready'
[restart]
prime c2: prompt=58673 cached=0 61.0s 'ready'
[restart]
```

Then, on a **fourth** fresh restart (RAM empty, FS tier now holds all
three, still zero conversations have ever coexisted in the same process),
sent turn 2 for all three — which the exposure log confirms was a genuine
full restore for every one of them:

```
KV LOAD chatcmpl-855ef45a63753bb3-8c303077 | fs -> RAM | 280 blk; hit 58240 offloaded tokens after 0 GPU hit tokens; RAM -> VRAM | 70 blk / 58240 tok (GPU had 0 tok)
KV LOAD chatcmpl-bc25b64eed1ef2ba-9923694e | fs -> RAM | 280 blk; hit 58240 ...; RAM -> VRAM | 70 blk / 58240 tok (GPU had 0 tok)
KV LOAD chatcmpl-90954cba53cbf2a4-9a2f874c | fs -> RAM | 280 blk; hit 58240 ...; RAM -> VRAM | 70 blk / 58240 tok (GPU had 0 tok)

c0 turn2: prompt=58693 cached=0 7.7s 'ok'
c1 turn2: prompt=58696 cached=0 6.9s 'ok'
c2 turn2: prompt=58697 cached=0 6.7s 'ok'
```

**All three clean.** `c1` — which corrupts in *every single* run of this
investigation where it was primed alongside other conversations — restores
perfectly when its own priming never shared a process with anything else.
Same 58,240-token boundary, same "0 GPU hit" full-restore exposure, same
content. Only the priming history differs.

**Conclusion: store-time contention (other conversations having shared the
process's memory before/around this chunk's capture) is necessary for the
corruption.** Isolation is sufficient to prevent it. This independently
confirms H-STORE and additionally rules out the "pressure framing is wrong"
alternative the plan flagged — the pressure framing is right.

## Phase B

**Not done as originally scoped** (the planned per-chunk EVICT/REALLOC
timeline table). A manual pass at it (grepping `EVICT block_id=` for `c1`'s
newest-chunk store job's 4 GPU block ids, job=139, and comparing against
`c0`'s job=69) found eviction events both immediately before job creation
(ordinary same-request block churn — expected, not suspicious) and much
later (60s+ after, safely post-completion) for both `c0` and `c1` alike, no
qualitative difference — inconclusive by inspection, and given A5 already
settled the necessity question more cleanly than a timeline table would,
this was not pursued further. If revisited, do it programmatically (parse
all EVICT/REALLOC/FLUSH lines into a real timeline, not by eyeballing grep
output) rather than by hand.

## Phase C — H-STORE branch (partial; methodological correction found)

**Attempted:** cross-run CRC reference comparison per the plan's item 1 —
capture `c1`'s STORE_DONE crc for its newest chunk (job, all 65
layers/groups) under isolated priming (checksums on, otherwise identical to
A5's recipe), then again under contended priming (the corrupting baseline,
checksums on), and diff.

**Result:** 48/65 mismatched (all 3 Mamba groups; the 17 "matches" turned
out to be a false signal, addressed below).

**This result is invalid as evidence.** Control test: ran the *isolated*
(clean-restoring) priming **twice** and diffed `c1`'s own STORE_DONE crcs
against itself, same recipe, same content, both runs known-clean:

```
tidx=0 gpu_blk=64: 837763a3 vs 837763a3  MATCH
tidx=0 gpu_blk=65: 76165473 vs 7245a879  DIFFER
tidx=0 gpu_blk=67: 403853c8 vs 98009f65  DIFFER
tidx=1 gpu_blk=64: 7245a879 vs 0d704106  DIFFER
...
matches: 5/65
```

Two demonstrably-clean, byte-identical-content runs produce almost entirely
different CRCs from each other. **GPU kernel execution for this Mamba/GDN
state is not bit-reproducible across separate process launches** (floating-
point reduction-order jitter, most likely — CUDA graph capture, kernel
scheduling, and memory layout all differ per process even with identical
inputs and `PYTHONHASHSEED=0`/temperature 0, which only guarantee
*token-level* determinism, not bit-level tensor determinism). The isolated-
vs-contended CRC mismatch reported above is consistent with ordinary
run-to-run jitter and **is not evidence of anything** — cross-run raw-byte
comparison is the wrong tool for this question and should not be reused
as-is.

**What survives this correction:** the *within-run* comparison from session
1 (store-completion → load-submission → GPU-read, all within one
continuous process, comparing physically-copied bytes rather than
independently-recomputed ones) is **not** subject to this jitter problem —
nothing recomputes between those three points, so a=b=c there is a real,
valid statement: RAM residency and the load pipeline are byte-faithful, for
both a clean and a corrupted restore. Combined with A5, the logical
conclusion is sharper than the original plan's H-STORE framing suggested:
the wrong bytes are not "correct bytes later corrupted by a reuse race" —
they are what the forward computation *itself* produces, at capture time,
when other conversations have shared the pool. A real computational
divergence, not a data-movement one. The "drift detector" (crc GPU-side
bytes at store-job *creation* vs at copy *completion*, both within the same
run) from the plan's Phase C item 2 was not attempted this session but
remains valid as a technique (same-run, not cross-run) and is the natural
next step if a specific block-mutation-mid-flight race is still suspected.

The attention-group GPU-read checksum (plan's H-RESTORE branch item 1) was
**not implemented** — A1/A5 already point at store-time/computation, not
restore-time/attention, as the locus, so this was deprioritized.

## Fix

Not implemented — root cause (which specific computation goes wrong under
contention) not yet localized to a mechanism, only to a phase (priming,
under contention) and a boundary (the newest complete Mamba/GDN chunk,
58,240 tokens here). No workaround-as-fix proposed.

## Anomalies & open questions

**Updated elimination table** (additions/corrections from this session; see
`RESULTS-restore-corruption.md` for the rest, which stands):

| candidate cause | status | evidence |
|---|---|---|
| RAM-tier / load-pipeline data-movement bug (session 1's a=b=c) | **ELIMINATED, reinterpreted** | still true, but now understood as "nothing recomputes in this window" rather than "content is correct" |
| Cross-run raw-byte / CRC comparison as a technique for judging correctness | **INVALID** | isolated-vs-isolated control: 5/65 match on two known-clean runs |
| Restore-side eviction/reuse race corrupting already-correct bytes | **DISFAVORED** | A5: isolated priming (impossible to race against nothing) is sufficient for clean restores; the "correct bytes, then raced" framing has nothing to race against in the isolated case, yet still needs *contention*, specifically, to reproduce — points at computation, not data movement |
| Forward computation itself diverges when other conversations have shared the pool | **LEADING, not yet localized to a mechanism** | A1 + A5 jointly; not yet reproduced with instrumentation pinpointing *which* per-request or per-slot state is contention-dependent |

**Sharpest remaining hypothesis:** some piece of state involved in computing
or capturing the Mamba/GDN boundary chunk is either (a) not properly reset
when a batch slot, block position, or similar resource that a *different,
now-finished* conversation previously used gets reused by the next one, or
(b) genuinely, correctly contention-dependent in a way this deployment's
Mamba/GDN chunked-prefill implementation doesn't handle losslessly (e.g. a
chunking-schedule difference under contention that a mathematically-correct
implementation should be invariant to, but isn't). A1/A5 cannot distinguish
these from each other — both predict "isolated clean, contended corrupt."

**Deciding experiment for a follow-up session:**

1. **Chunking schedule — CHECKED, ruled out.** Compared `c1`'s exact
   prefill step sequence (`prompt processing [...] (1 req, N tokens)` log
   lines) between the isolated run (`qwen.log.plan2-A5-prime-c1`) and the
   contended run (`qwen.log.plan2-C-contended-checksums`, req
   `chatcmpl-92cc13269fd6f7ed-ba1ddc4e`): **byte-for-byte identical** in
   both — 70 steps of 832 tokens followed by one 432-token step, same
   order, same sizes. Whatever differs between isolated and contended
   priming, it is not *when* or *how much* gets scheduled per step.
2. **Next, and now the leading candidate:** the same-run "drift detector"
   from plan 2's Phase C item 2 — crc the GPU src bytes at store-job
   *creation* time (scheduler decision) and again at copy *completion*
   time, both within one process, for `c1`'s boundary chunk in a contended
   run. A difference here (immune to the cross-run jitter problem, since
   both hashes are from the same physical bytes at two points in the same
   execution) would directly prove something mutates the block between
   "the scheduler decided this chunk is final" and "the copy actually read
   it." If that comes back clean too, the divergence must be even earlier —
   inside the forward pass itself, at the point the chunk's tokens are
   actually computed — which would mean auditing whatever *is* legitimately
   different about the compute graph/kernel selection when other
   conversations have previously occupied the same physical GPU blocks
   (padding, block-table contents for now-finished-but-slot-reused
   requests, CUDA-graph bucket selection) even though the batch itself is
   always size 1 at the moment `c1` runs (this test's HTTP client is
   strictly sequential/blocking — never true concurrency, so whatever the
   mechanism is, it does not require concurrent admission, only *prior*
   occupancy).

**Housekeeping:** server left running in default production config (no
debug env vars). The Phase-1 checksum debug patch
(`patches/restore-corruption-checksum-debug.patch`) is unchanged and still
applied (safe, off by default). Numerous `qwen.log.*`/`qwen.log.2026-*`
snapshots from this session are on disk in the repo root, not committed
(large, `.gitignore` already excludes `qwen.log` itself) — the load-bearing
ones for this doc's claims are named `plan2-A1-*`, `plan2-A5-*`, and
`plan2-C-*`; everything else from this session is superseded and safe to
delete once this doc's excerpts are trusted.
