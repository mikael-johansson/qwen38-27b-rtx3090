#!/bin/sh
# Vision variant of micke-start.sh. Identical production config, plus:
#   VISION=1                     build the ViT tower (drops --language-model-only)
#   VISION_OFFLOAD=prefetch      stream its 27 blocks from pinned host RAM
#   --limit-mm-per-prompt        REQUIRED with VISION=1 -- the 999-per-modality
#                                default makes startup memory profiling reserve
#                                an absurd activation peak
#   --mm-processor-kwargs        cap at 1024x1024; overhead scales with patch count
#
# Needs patches/vision-tower-cpu-offload.patch applied to the venv; start_qwen.sh
# refuses to launch otherwise rather than silently loading the tower resident.
#
#   VISION_OFFLOAD=off MAX_LEN=130000 sh micke-start-vision.sh   # A/B baseline
#
# micke-start.sh (text-only) is deliberately left untouched; see
# VISION_TOWER_SWAP_PLAN.md.
ITERATION_LOG=0 \
REQUEST_LOG_DIR=$(pwd)/requests \
VLLM_LOG_STATS_INTERVAL=10 \
VLLM_CACHE_METRICS_WINDOW=3 \
PREFIX_CACHE=1 \
CTX=long \
MAX_LEN=${MAX_LEN:-140000} \
MAX_SEQS=${MAX_SEQS:-4} \
KV_MEM=${KV_MEM:-5606637568} \
PYTHONHASHSEED=0 \
VLLM_OFFLOAD_EAGLE_FALLBACK=0 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
VISION=1 \
VISION_OFFLOAD=${VISION_OFFLOAD:-prefetch} \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --long-prefill-token-threshold 832 --limit-mm-per-prompt {"image":{"count":2,"width":1024,"height":1024},"video":0} --mm-processor-kwargs {"max_pixels":1048576} --mm-processor-cache-gb 2 --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":21474836480,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv","max_disk_gib":200}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
#
# --- co-tenancy with whisper.cpp STT (2026-08-27) -------------------------
#
# KV_MEM=5606637568 (5347 MiB) pins the KV pool in BYTES instead of deriving it
# from GPU_UTIL. gotcha 18 recommends this: the profiled activation peak varies
# by ~1 GiB between starts of the same config, so a utilization-derived pool is
# not reproducible. Pinning also makes the co-tenancy arithmetic exact.
#
# Measured on this box:
#   vLLM footprint = 16330 MiB (weights + graphs + activations, constant)
#                  + KV_MEM
#   whisper.cpp STT server        ~632-724 MiB (grows a little with use)
#   DeltaNet/GDN prefill spike    ~1208 MiB (2 concurrent reqs, 40k prefill)
#
#   16330 + 5347 + 724 + 1208 = 23609 of 24576  ->  ~967 MiB spare at peak,
#   which matches the headroom the old whisper-less 0.93 config ran with.
#
# This yields 146,847 tokens (1.05x at MAX_LEN=140000). To trade margin for
# pool: KV_MEM_MiB = 24576 - 16330 - <other resident> - 1208 - <margin>.
# Set KV_MEM= (empty) to fall back to GPU_UTIL sizing.
#
# GPU_UTIL stays 0.93; with KV_MEM set it only bounds the budget, it no longer
# determines the pool.
#
# MAX_SEQS=4 (was 8): cuts the transient peak ~288 MiB under concurrent load
# for 0.5% of the pool. Single-user needs 4 slots.
#
# START ORDER DOES NOT MATTER (verified both ways, identical pool).
#
# TWO THINGS THAT DO BREAK STARTUP, both from a previous unclean exit:
#   1. An orphaned VLLM::EngineCore still holding ~16 GiB:
#        nvidia-smi --query-compute-apps=pid,used_memory --format=csv ; kill <pid>
#      Then WAIT for the GPU to actually drain (gotcha 2) -- the systemd units
#      gate on `memory.used < 1000` for up to 120 s. A bare `sleep 5` is not
#      enough and produces a silently undersized pool, or an OOM at model load.
#   2. An orphaned 20 GiB /dev/shm segment from the KV-offload CPU tier, which
#      makes the next start die with `OSError: [Errno 14] Bad address` in
#      kv_offload/cpu/shared_offload_region.py -- nothing to do with VRAM:
#        ls -la /dev/shm/vllm_offload_*.mmap ; rm -f /dev/shm/vllm_offload_*.mmap
#      (check `fuser` first that no live engine holds it).
