#!/bin/sh
ITERATION_LOG=1 \
REQUEST_LOG_DIR=$(pwd)/requests \
VLLM_LOG_STATS_INTERVAL=10 \
VLLM_CACHE_METRICS_WINDOW=3 \
PREFIX_CACHE=1 \
CTX=long \
MAX_LEN=140000 \
PYTHONHASHSEED=0 \
VLLM_OFFLOAD_EAGLE_FALLBACK=0 \
VISION=1 \
VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-0} \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file.json \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --long-prefill-token-threshold 832 --limit-mm-per-prompt {"image":{"count":2,"width":1280,"height":1280},"video":0} --mm-processor-cache-gb 2 --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":21474836480,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv","max_disk_gib":200}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
# PYTHONHASHSEED=0 -- fixes the KV-block hash chain's seed (NONE_HASH in
# vllm/v1/core/kv_cache_utils.py) to a constant instead of os.urandom(32),
# which vLLM otherwise regenerates fresh on every process start. Without
# this, every block hash for the whole session comes out different across
# restarts even for byte-identical prompts, so the NVMe fs-offload tier's
# on-disk cache (patches/fs-tier-disk-cap.patch) is unreachable after a
# restart -- the files are still there, just under hash-paths this new
# process can never re-derive. With a fixed seed, a restarted server can
# actually find and reuse its own prior session's offloaded blocks. See
# 2026-08-22 conversation. (0 is an arbitrary fixed constant, not
# security-sensitive here -- single-user, trusted deployment.)
#
# VISION=1 + the four multimodal flags -- loads the vision tower (2026-08-23).
# Qwen3.5 is a VL model (Qwen3_5ForConditionalGeneration) and this quantized
# checkpoint already ships all 333 model.visual.* tensors, 879 MiB BF16
# (deliberately excluded from the W4A16 quant, as vision towers usually are).
# Until now --language-model-only kept limit_mm_per_prompt=0, which makes
# vLLM's _mark_tower_model skip instantiating the tower entirely -- so it cost
# no VRAM but images were simply unsupported.
#
#   VLLM_VISION_CPU_OFFLOAD_GB (patches/vision-tower-cpu-offload.patch)
# Default 0 = ViT weights stay in VRAM. Set to 1 to move the 27 ViT blocks
# (0.77 GiB) into pinned host RAM, read over PCIe via UVA -- compute still runs
# on the GPU. MEASURED TRADE on this box (bench/vision_offload_ab.py):
#     KV pool   0 -> 5.17 GiB / 145,326 tok      1 -> 5.95 GiB / 167,391 tok
#     image encode latency: 2.6-3.8x SLOWER when offloaded
#       224px 0.19->0.49s   448px 0.46->1.29s   896px 0.92->2.54s
#       1280px 1.34->5.03s
# Left OFF by default because a 2.6-3.8x hit on every image is very visible,
# while the ~29 blocks it buys back matter only under heavy concurrency. Flip
# to 1 (or a partial budget like 0.4) if KV blocks turn out to be worth more
# than image latency for your traffic. The cost scales with patch count, so
# lowering --limit-mm-per-prompt's width/height also bounds it.
#   Why it is not just "one PCIe pass over 0.77 GiB (~40 ms)": GEMM kernels
# tile, so each weight tile is re-read once per row-block of the activation
# matrix -- many passes, not one. Cheap from VRAM (~936 GB/s), dominant across
# PCIe (~20 GB/s). UVA offload suits memory-bound layers, not a compute-heavy
# ViT. Full write-up in the patch header.
#
# NOTE the stock flags cannot do this. `--cpu-offload-gb 1 --cpu-offload-params
# visual --offload-backend uva` is accepted without error but is a silent NO-OP
# for vision towers (tried 2026-08-23, removed):
#   * vllm/model_executor/offloader/uva.py implements exactly what we wanted
#     (weights in pinned host RAM, mapped into the GPU address space via UVA,
#     read over PCIe during the forward; compute stays on GPU, CPU never runs
#     the ViT) -- and is_uva_available() is True on this box.
#   * "visual" is also segment-exact (the matcher is
#     `f".{param}." in f".{name}."`): verified it matches exactly the 333
#     model.visual.* tensors and zero language-model tensors.
#   * BUT the offloader's only call site in the whole codebase is
#     `get_offloader().wrap_modules(...)` inside make_layers()
#     (model_executor/models/utils.py:824), and make_layers() builds only the
#     *language model* decoder stack (qwen3_5.py:257). The vision tower builds
#     its blocks with a plain `nn.ModuleList` (qwen3_vl.py:628), so the
#     offloader never sees them. Confirmed: no "Total CPU offloaded
#     parameters" line is logged, and the KV pool shrinks by the full tower.
# That is what patches/vision-tower-cpu-offload.patch fixes, by routing
# Qwen3_VisionTransformer.blocks through a dedicated UVAOffloader --
# VLLM_VISION_CPU_OFFLOAD_GB above. It works; it is just not worth it by
# default at the measured latency cost.
#
# Net effect at the shipped default (VLLM_VISION_CPU_OFFLOAD_GB=0): the vision
# tower lives in VRAM and costs ~0.94 GiB of KV pool --
# 6.11 GiB / 171,956 tokens / 225 blocks  ->  5.17 GiB / 145,326 tokens / ~190
# blocks. A single full-length request still fits (145,326 > MAX_LEN 140,000);
# what shrinks is concurrency headroom, so watch for preemptions under heavy
# multi-conversation load and consider MAX_SEQS=6 if they appear. VISION=0 gets
# the whole pool back (tower not instantiated at all).
#
#   --limit-mm-per-prompt {"image":{"count":2,"width":1280,"height":1280},"video":0}
# NOT optional. The default is 999 items per modality, and startup memory
# profiling reserves the worst-case activation peak for that -- which would eat
# the KV pool. count/width/height bound both the profiling reservation and the
# per-image vision-token count (1280x1280 at patch_size 16 with spatial_merge 2
# => ~1600 vision tokens). Raise deliberately, and re-check the resulting
# "GPU KV cache size" log line if you do.
#
#   --mm-processor-cache-gb 2
# The default is 4 GiB and it is duplicated per API-server and engine-core
# process (= 8 GiB of HOST RAM at api_server_count=1). This box has 31 GiB with
# 20 GiB already committed to the CPU KV tier's /dev/shm mmap.
#
# Deliberately NOT set: --mm-encoder-attn-dtype fp8 (FP8 ViT attention). It
# would save activation memory but changes vision numerics; get correctness
# established first, then treat it as a separate, measured change.
#
# VLLM_LOGGING_CONFIG_PATH -> logging-to-file.json (INFO), switched back from
# logging-to-file-debug.json on 2026-08-23 now that the offload/HIT_DIVERGED
# investigation is closed and patches/offload-verbose-eviction-debug-logging.patch
# is reverted (it was ~75% of all log lines: 324k of ~430k in a 55-min window,
# ~100 MB/hour). The one line worth keeping in production -- "HIT_DIVERGED
# RECONCILE", i.e. real work was thrown away -- was promoted from debug to
# WARNING so it survives at INFO. Re-apply that patch and flip this back to
# logging-to-file-debug.json if the offload path ever needs deep debugging again.
#
# --long-prefill-token-threshold 832 -- caps how many tokens of a single
# request's prefill get scheduled per step, even once that request is
# already admitted into the running batch. Without this (default 0 =
# unlimited), an admitted long prompt is entitled to the entire
# --max-num-batched-tokens (2048) budget every single step until its own
# prefill finishes, starving other running/waiting requests of that step's
# budget the whole time -- confirmed by reading scheduler.py's schedule():
# the running-phase loop runs before the waiting-phase loop and hands out
# token_budget FCFS with no per-request cap unless this is set. A big prefill
# now yields the rest of the budget back every step, letting other
# conversations' decode and other prefills' chunks interleave instead of one
# long prompt monopolizing the server until it's done. See 2026-08-22/23.
#   832, not 512: 832 is this model's mamba block size, and the align-mode
# block-aligned split clips every chunk to a block boundary. At 512 that split
# produced an alternating 512+320 -- TWO scheduler steps per 832-token block
# (measured: 1491 x 512 and 1412 x 320 chunks in one window), costing ~11.5%
# prefill throughput (median 875 -> 774 tok/s on solo prefills >= 4000 new
# tokens). At 832 it is exactly one block per step. It also still leaves room
# for a second concurrent prefill to complete a full block in the same step
# (832 + 832 = 1664 <= 2048), which the uncapped 1664 does not.
#
# Note: vLLM also has --scheduling-policy priority (default: fcfs), which
# would let requests carry an explicit priority (the OpenAI-compatible API
# already accepts a `priority` field per request, currently ignored under
# fcfs) -- lower-priority *admission* order and, if KV blocks are scarce,
# actual preemption of the lowest-priority running request. Not enabling
# this now (fcfs is fine given long-prefill-token-threshold above already
# addresses the main starvation complaint) -- worth revisiting if we want
# specific requests (e.g. short agentic deltas) to jump the waiting queue
# ahead of a big prefill rather than just interleaving with it. Note it
# does NOT bypass the --max-num-seqs=8 admission-slot cap either way.
# VLLM_OFFLOAD_EAGLE_FALLBACK=0 -- disables the offload connector's
# "mark every KV group as an EAGLE/MTP draft group" fallback
# (patches/offload-eagle-misclassification-mamba.patch adds this knob;
# default 1 keeps upstream behavior). This model's MTP drafter has one
# attention layer (mtp.layers.0.self_attn.attn) sharing the full-attention
# KV group, so the fallback marks that group as draft-volatile: its lookup
# pops one provisional trailing chunk, and its newest complete chunk is
# excluded from storing during decode. On turn N+1 of a conversation whose
# shared prefix is still GPU-resident the external hit beyond the local
# boundary is typically exactly that 1 popped chunk, so the external hit
# collapses to 0 and the scheduler's HIT_DIVERGED reconcile throws away a
# 30-50K-token GPU-resident prefix for a full reprefill (traced 2026-08-23,
# two events in qwen.log.evidence-mamba-free-never-fired.log; 29 events /
# 825K tokens discarded in one 20-min 3-way run). Worst case of trusting
# the trailing chunk is a briefly-stale draft-layer KV near a resume
# boundary => slightly lower spec-decode acceptance there; MTP drafts are
# always verified by the target model, so output correctness is unaffected.
# See RESULTS-hit-diverged.md (2026-08-23) for the measured comparison.
# max_disk_gib:200 on the fs secondary tier -- patches/fs-tier-disk-cap.patch
# adds this parameter to vLLM's FileSystemTierManager (no upstream equivalent
# exists in 0.27.1, unlike the primary CPU tier's cpu_bytes_to_use). A
# background thread inside the connector itself checks every 2 minutes and
# evicts the least-recently-accessed *.bin block files (LRU by atime) once
# usage exceeds 200GiB -- runs continuously for the life of the server, not
# just at startup, so it actually keeps up with a long-running session
# (the earlier prune_kv_offload_cache.sh external-script approach only
# caught growth between restarts, which isn't enough at 200GB/hours-to-days
# fill rates -- superseded by this, kept in the repo as a manual/one-off
# tool only). See docs/mamba-align-prefill-leak.md's "no disk cap" section.
# kv-transfer-config above replaces --kv-offloading-size, with cpu_bytes_to_use
# at 20GB (host has 32GB total; ~5GB non-offload baseline measured directly).
# Was 24GB literal, but that measured baseline undercounted real-world
# pressure -- two concurrent ~88K-token prompts pushed vLLM's own processes
# into swap at 24GB (verified via /proc/<pid>/status VmSwap), so backed off
# by 4GB for a firmer safety margin -- accepted tradeoff, see 2026-08-21
# conversation), plus a "fs" secondary tier so blocks
# evicted from RAM cascade to NVMe (/d/nvme_cache/vllm_kv, 194GB free) instead
# of being lost. The JSON has NO
# spaces deliberately -- EXTRA_ARGS is expanded unquoted in start_qwen.sh,
# so any space would split it into multiple argv words and corrupt the
# JSON; verified this survives word-splitting intact before using it here.
# kv_role=kv_both is required by vLLM's KVTransferConfig validator whenever
# kv_connector is supplied explicitly (unlike --kv-offloading-size's default
# path, which sets kv_connector via attribute assignment after construction
# and so never triggers that validator) -- confirmed by direct testing.
#
# VLLM_LOGGING_CONFIG_PATH -> logging-to-file-debug.json (not the normal
# logging-to-file.json): temporary, for the preemption/offload-cascade
# investigation (patches/preemption-offload-debug-logging.patch). File
# handler + "vllm" logger at DEBUG so the "request-logging preemption-
# debug: ..." trace lines land in qwen.log; console handler stays INFO so
# the terminal isn't flooded with every debug line in vLLM. Switch back to
# logging-to-file.json once this investigation is done.
#
# --watermark: tried at 0.35 and REVERTED (2026-08-21). Tested against a
# single unopposed ~88K-token prompt (the second request correctly parked
# in queue, qwait=74.7s, never raced it) -- and the first one *still* took
# 433.89s (vs 435.955s without watermark), self-preempting 3 times purely
# on its own. watermark applies the same headroom check to a request's own
# PREEMPTED-status retry as it does to a genuinely new competing request
# (kv_cache_manager.py can't tell them apart), so for a single prompt this
# size (~88K, roughly half the ~172K pool) the watermark reserve competes
# with the request's own resumption instead of only blocking a rival. Net:
# the original "two requests racing" framing was incomplete -- a single
# request this size self-preempts repeatedly even with zero contention, so
# admission-control against a second request was never the fix for the
# bulk of the observed latency. See 2026-08-21 conversation.
