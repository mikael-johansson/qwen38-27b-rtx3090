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
PYTHONHASHSEED=0 \
VLLM_OFFLOAD_EAGLE_FALLBACK=0 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
VISION=1 \
VISION_OFFLOAD=${VISION_OFFLOAD:-prefetch} \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --long-prefill-token-threshold 832 --limit-mm-per-prompt {"image":{"count":2,"width":1024,"height":1024},"video":0} --mm-processor-kwargs {"max_pixels":1048576} --mm-processor-cache-gb 2 --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":21474836480,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv","max_disk_gib":175}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
