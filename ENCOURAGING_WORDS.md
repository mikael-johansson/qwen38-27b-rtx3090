# Encouraging words
Are we on the right track? Are we pursuing something valuable here? Apparently, yes.

### Me
Ok. Quick followup while we're running the plan: Is mamba_cache_mode=align something I need for my use case? What I want is to run this optimized repo on my memory-constrained RTX3090, without ever having to _recalculate_ prefill blocks when they are evicted. I want a working VRAM --> RAM --> nVME offloading pipeline (With working restoration too, of course). Beyond that, I want as much concurrency / performance as I can fit in the tight KV cache. If possible, I also want to fit the vision tower (Sacrificing some KV cache / concurrency). Am I using the right parameters or am I making this more difficult than it needs to be?

### Fable 5
Short answer: yes, align is load-bearing for exactly the thing you want, and no, you're not over-complicating it — the complexity is intrinsic to your model choice, not your flags.

Why align is required for your goal. Your model is a hybrid: 48 of 64 layers are GDN/linear-attention with a single evolving recurrent state instead of per-token KV. "Never recompute prefill" requires that a later request (turn N+1, or a restore after eviction) can resume from stored state — and a recurrent state is only resumable if it was checkpointed at a known position and is findable by content. That's precisely what align mode is: it snapshots the recurrent state at 832-token block boundaries and keys those snapshots by the same content-hash chain as attention blocks, which is what makes them prefix-cacheable, offloadable, and restorable. Without it (mode all), mamba state is a per-request mutable blob: no cross-turn reuse, no offload lookup, every evicted or new-turn prefix is a full recompute. Your entire pipeline goal dies at the first linear-attention layer.

**Worth saying: the model choice is also why your use case is feasible at all. This hybrid runs ~35KB/token of KV+state; a dense 64-layer model with the same head config would be ~4x that — your 172K-token pool would shrink to ~45K, and 2×80K concurrent conversations would be physically impossible on a 3090. So you're not paying the align-mode tax for nothing; it's the price of the context capacity that makes the whole setup work. The flip side is that align + OffloadingConnector + tiering is a near-unique configuration — which is why you've personally found six real upstream bugs in it. That's the frontier, not misconfiguration.**

The parameter set itself is broadly right. PREFIX_CACHE/align, the CPU+fs tiers, PYTHONHASHSEED=0, max_disk_gib, --long-prefill-token-threshold 832 (aligning the scheduling chunk to the mamba block is genuinely a good idea), batched-tokens 2048 — all consistent with the goal. Two knobs are optional complexity you could shed if you value stability over speed while the bug tail is being worked through:

- MTP spec decode is the biggest interaction-surface multiplier (the eagle guard, the optimistic-storability defect, accepted-token-bias state copies all exist because of it). Dropping it costs maybe 30–40% decode throughput and removes a whole bug class. I'd consider parking it until the restore bug is fixed and gates pass, then re-enabling.
- --async-scheduling is the second — it's implicated in the one unexplained CUDA crash and complicates every timing argument. Same logic: it's a performance knob, not a correctness need.

Vision tower: it's a straight VRAM-for-capability trade and composes fine with the rest — image embeddings become ordinary tokens whose KV lives in the same cache and offloads the same way. But on this card it comes directly out of your KV pool (concurrency and max resumable context shrink accordingly), so I'd sequence it strictly after the restore bug is closed and the gates are green, so you're never debugging two capacity regimes at once.

Bottom line: keep align, keep the pipeline; the architecture already demonstrated your target behavior (98% cumulative hit rate, zero preemptions) when the bugs weren't biting. If you want a faster route to boring reliability, the levers are dropping MTP and async-scheduling temporarily — not abandoning align, which would abandon the goal itself.
