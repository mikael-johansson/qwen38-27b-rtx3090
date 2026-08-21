# Gotchas

Things that each cost us hours, in rough order of pain. Worth skimming before you debug something that looks like a vLLM bug.

[← back to the main README](../README.md)

1. **A benchmark cannot tell you the output is garbage.** The int8-activation
   path served nonsense for an hour of beautiful throughput numbers before a
   perplexity check caught it. Whatever you change, run
   `bench/quality_battery.py` (perplexity + GSM8K against the live server)
   before you believe a tok/s number.
2. **Restart onto a dirty GPU and you silently lose 25%.** vLLM profiles free
   memory once at startup. If the previous process is still releasing VRAM at
   that moment, the cache pool comes out ~40% smaller and stays that way. No
   warning, the server runs fine, throughput is just quietly bad. The systemd
   units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU
   to be actually free.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional.** The
   DeltaNet prefill kernels allocate transient workspace; without it the
   allocator fragments and the engine OOMs at runtime once
   `gpu-memory-utilization` goes past ~0.975.
4. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.93`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you. We soak-tested 0.93 with a
   100k-token prompt plus 6k-token generations at 4 concurrent.
5. **The torch.compile cache does not know about your env vars.** Switching
   `INT8_LAYERS` between runs replays a compiled graph that expects the other
   layer set and dies with `KeyError: 'input_global_scale'`. Our patch
   registers the selection env vars with vLLM so they become part of the cache
   key; if you invent your own, `VLLM_DISABLE_COMPILE_CACHE=1`.
6. **Random-token benchmarks flatter speculative decoding.** See the ninfer
   section: the same server does 35, 83 or 151 tok/s on `--dataset-name random`
   depending on what the noise turns into. Use real prompts
   (`--dataset-name custom`).
7. **Bigger prefill chunks make things worse — worse than this entry used
   to say.** `--max-num-batched-tokens 8192` inflates the profiled
   activation peak (shrinks the cache pool, caps concurrency) *and*, on
   `--mamba-cache-mode align`, larger chunks hit a real vLLM leak in the
   Mamba/GDN state-freeing check that never frees state during a
   continuous prefill — a single, unopposed large prompt can self-preempt
   against nothing but its own leak, or OOM-crash the engine outright at
   8192. 1024 wins on this card, not 2048; see
   [docs/mamba-align-prefill-leak.md](mamba-align-prefill-leak.md) for the
   full mechanism and what the real upstream fix would need to touch.
8. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low.
9. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded). If you don't need images, that's 2.7 GB.
10. **`prompt_logprobs` on long prompts OOMs the engine at 0.972 utilization**
    (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom).
    Run quality checks at 0.93.
11. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
    the decode kernel across block/warp configs: it already runs at ~85% of the
    3090's memory bandwidth and every variant lands within 3%. The state dtype
    (point 3 above) is the lever, not the kernel.

12. **The draft vocabulary is the single-user ceiling.** A draft head can only
    propose tokens in its id list; a miss is a certain rejection that also ends
    the chain. Count the list over the model's *own* outputs (`drafter/gen_data.py`,
    then the frequency step in `build_draft_vocab.py`), not over web text —
    92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy.
    Coverage saturates around 40k rows; the model only ever emits ~54k distinct
    tokens.
13. **FlashAttention-2 does not split KV for multi-query decode.** With k
    speculative tokens the verify step has k+1 queries per request and FA2's
    varlen path then runs one thread block per (request, head): 24 blocks on 82
    SMs, 57 µs per layer at 1.5k context and 1.3 ms at 16k. vLLM's Triton
    unified attention has the same restriction (`max_seqlen_q > 1` → 2-D
    kernel). `patches/spec-decode-attn.patch` (`VLLM_SPEC_DECODE_ATTN=1`, bf16
    KV only) is a 180-line Triton fix. Watch its query cap: the kernel used to
    handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall
    back silently past that, which doubled the step at 25k context the moment
    the verify block grew to 16. It now tiles the query rows instead.
14. **Greedy is not deterministic across drafter configs.** The target rounds
    differently when it verifies 5 tokens vs 1, so a different drafter changes
    the generated text at near-ties and the 8-prompt acceptance numbers move
    ±3%. Repeat before trusting a small difference; `drafter/README.md` has an
    offline chain simulator that removes the noise.
15. **A stale torch.compile cache bites anything that changes tensor shapes
    behind vLLM's back.** The compiled graph bakes in e.g. the Marlin workspace
    size; a new env knob that changes it must be registered in `envs.py`
    (`patches/speed-knobs-envs.patch`) or you get `assert_size_stride ...
    expected size 328==82` from a cached artifact.
16. **The very first start gets a smaller KV pool.** vLLM sizes the pool from
    the peak memory of a profiling forward pass, and on a cold torch.compile
    cache that pass also runs inductor's autotuning: batch mode profiles a
    1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV
    tokens instead of 224k (`Maximum concurrency ... 1.31x` in the log instead
    of 1.49x). Restart once after the cache is warm (venv: `~/.cache/vllm`,
    Docker: the `qwen-cache` volume) and the pool is back to the README numbers.
    (The WSL2 notes above pin it the other way round — record the cold-start
    `--kv-cache-memory` recommendation and pass it via `EXTRA_ARGS` — if you
    prefer the extra transient headroom to the extra KV pages.)
17. **vLLM picks the speculative method from the model *path*.** `"dflash" in
    model_path` switches `method` to dflash — for the *target* too, since MTP
    uses the target path as its draft model. A checkout under a directory with
    "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig`
    (`'Qwen3_5Config' object has no attribute 'vocab_size'`). Name your
    directories accordingly.
18. **The V2 model runner (`SPEC=dflash2`) does not count its CUDA graphs when
    sizing the KV pool** (~1.2 GiB on top of whatever `--gpu-memory-utilization`
    you asked for), the hybrid allocator sizes KV groups by the smallest layer
    bucket, and the profiled activation peak varies by ~1 GiB between starts of
    the *same* config — three ways to get a server that either wastes a quarter
    of its pool or dies mid-request. `patches/hybrid-kv-groups-v2-cudagraph.patch`
    fixes the first two; for the third, pin the pool in bytes
    (`--kv-cache-memory`, what `KV_MEM` does) instead of tuning utilization. That
    runner also answers `thinking_token_budget` with 400, and the first request
    after a cold start JIT-compiles four Triton kernels (~5 s once; cached in
    `~/.triton`).
19. **`INT8_LAYERS=.` needs `GPU_UTIL=0.95`.** Quantizing the activations of every linear
    layer (rather than just the MLP) is worth ~11% throughput — 1,042 vs 942 tok/s at 64
    concurrent — but the extra per-layer scratch no longer fits batch mode's 0.972: the
    engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are
    resident, which reads as every request returning 500 while `/health` still answers.
20. **A Triton kernel's scratch buffers may not grow after CUDA graph capture.** The
    split-KV verify attention sizes its partial buffers from the longest query block it has
    been asked for. Once the block got longer than the drafter's — and once a small prefill
    chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers
    were reallocated, and the captured decode graph went on reading the freed ones:
    `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the
    first request. `VLLM_SPEC_DECODE_ATTN_QMAX` (set by `single-user/start_qwen.sh` from
    `DFLASH_TOKENS`) fixes the size at startup instead.
21. **Async scheduling pins the number of speculative tokens.** vLLM only feeds draft token
    ids — and therefore the *count* the worker wants verified — back to the scheduler on the
    synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is
    padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently.
    Adaptive block length (`LOOKUP=1` with `DFLASH_TOKENS > 7`) needs `ASYNC_SCHED=0`; at
    batch 1 that costs under 1%.
22. **`--async-scheduling` is already the default in 0.27.1.** The flag exists and passing it
    changes nothing; `--no-async-scheduling` is what turns it off. Two hours of "the adaptive
    block isn't working" was this.
23. **A longer verify block costs KV pool per request slot, not per token.**
    `--mamba-cache-mode align` reserves `2 + num_speculative_blocks` recurrent-state pages
    per slot, so `DFLASH_TOKENS=31` with 8 slots wants 5.3 GiB before a single token of
    context and refuses to start. Single-user mode drops to 4 slots when the block is long,
    which is what makes the long block affordable at all.
24. **The DFlash draft pass is a captured CUDA graph, so its Python runs once.**
    `DFlashSpeculator._generate_draft` — everything the speculator does per step, including
    the lookup — is replayed from a graph. The Triton kernels inside it do run every step and
    do read live buffers, so the lookup itself works; but host-side Python in there executes
    at *capture* time only. A counter, a pinned copy of a flag, a decision computed there is
    frozen at whatever the warm-up produced, silently. Anything the host must see per step
    belongs in a method the model runner calls per step (`next_num_draft_tokens`), reading
    device tensors the replayed kernels wrote. Three separate "the trigger doesn't fire"
    debugging rounds were this.
25. **`torch.cuda.is_current_stream_capturing()` is not a usable guard on this path.** It
    reads True inside the captured draft pass — which is correct, and exactly why a guard
    written as `if not is_current_stream_capturing():` silently disables the code it guards
    for the entire run, not just during warm-up.
26. **rsync preserves mtimes, and Python trusts mtimes.** Copying a source file into
    `site-packages` with `rsync -a` can leave the `.pyc` newer than the `.py`, in which case
    the interpreter keeps running the old bytecode and every measurement lands on the
    previous revision. Delete `__pycache__` after installing patched files.
27. **A shorter draft block than `num_speculative_tokens` loses the decode CUDA graphs.**
    The V2 runner captures uniform-decode graphs at `decode_query_len = num_speculative_tokens
    + 1` and dispatch requires an exact match, so scheduling the drafter's 8-token block on a
    16-token server matches nothing and the step runs piecewise: 27.9 ms against 25.9 ms for
    the same work, on every short step. `cudagraph_utils.py` already knows how to capture
    several decode lengths (it does it for dynamic speculative decoding); the lookup patch
    adds the drafter's block to that list. Costs 1.8 GiB of graphs instead of 1.45.
28. **A verify block costs step time in steps, not smoothly, and the two stairs are at 16
    and 21 query tokens.** Measured on a copy at 25k context: 39.5 ms per step at 16 query
    tokens (`DFLASH_TOKENS=15`), 47.8 at 19, 47.2 at 21 — a jump between 16 and 19 and then
    flat. The first stair is the target's W4A16 GEMMs: GPTQ-Marlin tiles the M dimension in
    16 rows (`m_block_size = 16 * thread_m_blocks`, `thread_m_blocks = div_ceil(prob_m,
    16)`), so a 17th query token buys a second M block in all 64 layers and the tokens up to
    32 are then free. The second is the verify attention: `SpecDecodeAttention._plan`
    (patches/spec-decode-attn.patch) puts `q_len * G` rows in a 128-row tile, so with this
    model's `G = 24/4 = 6` one tile holds `128 // 6 = 21` query tokens and a 22nd re-reads
    the request's whole KV segment (250/583/1132 us per layer at 8/16/32).
    So there are exactly two sensible block lengths — 16 query tokens, the last one on the
    bottom stair, and 21, the most tokens obtainable for the price of the second. 31 pays
    both stairs and was never worth measuring; two attempts to start it died on memory
    first.
29. **A verify block that outgrows its CUDA-graph reservation OOMs at run time, not at
    startup.** `--kv-cache-memory` pins the pool, so `VLLM_V2_CUDAGRAPH_MEM_MIB` no longer
    sizes it — it only reserves headroom, and if it under-reserves, the server starts, logs a
    healthy pool, and then dies on the first prefill with 50 MiB left. Graph memory grows
    with the block: measured 1.82 GiB at `DFLASH_TOKENS=15`, 2.12 at 18, 2.27 at 20 (the
    capture list length barely matters — 2.21 GiB at 20 with `CG` cut from 63 to 42). Budget
    a request as `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)`, the second term being
    the aligned recurrent-state pages, and take the extra graph memory out of the pool.
30. **The draft model is not redundant during a copy, even when the lookup overwrites every
    token it proposed.** It looks like free money: on a step the lookup controller selected,
    a qualifying match is long enough to take the head of the block too, so all seven of the
    drafter's tokens are replaced before anything is verified — skip its forward and save
    ~3 ms of a 39 ms step. Measured, that trade loses: 15.21 tokens per step becomes 13.79
    for a 5% cheaper step, a net 6% down. The drafter is covering the positions *past the
    end of the match*, which is exactly where a copy lands when the text it is reproducing
    diverges. Restricting the skip to steps where the match reaches the end of the block
    recovers the acceptance but only two runs in three — the flag it keys on is one step
    stale, and a stricter condition is more sensitive to that. Both variants are gone; this
    entry is here so the idea does not look untried.
31. **Any controller state that outlives one step has to be per-request, or batch > 1 stops
    being reproducible.** The lookup's block-length decision is batch-wide by design — a long
    block costs step time on every request in the batch — and taking it from the current
    step's flags is fine, because those are a function of the requests present. Holding it
    across steps is not: `VLLM_DFLASH2_LOOKUP_STICKY` keeps the long block on through steps
    where the flags say no, so with several requests in flight the block length a copying
    request gets depends on when the *others* arrived, and the block is one chunk through the
    recurrent layers, so that changes its greedy text. `bench/labd_soak.py` caught a verbatim
    copy coming out differently in two rounds of an identical four-way batch, and OK in three
    of three with the hold off. It is now applied only with one request in flight. The proper
    fix is per-request draft counts, which `get_uniform_token_count` in
    `gpu/cudagraph_utils.py` will not dispatch a graph for — a ragged batch runs piecewise
    and costs 8%, more than the hold is worth.
