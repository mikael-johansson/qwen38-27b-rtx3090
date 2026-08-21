#!/bin/sh
ITERATION_LOG=1 \
REQUEST_LOG_DIR=$(pwd)/requests \
VLLM_LOG_STATS_INTERVAL=1 \
PREFIX_CACHE=1 \
CTX=long \
MAX_LEN=140000 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file.json \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":12884901888,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv"}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
# kv-transfer-config above replaces --kv-offloading-size: same 12GB CPU-RAM
# primary tier (cpu_bytes_to_use is the byte equivalent of that flag), plus
# a "fs" secondary tier so blocks evicted from RAM cascade to NVMe
# (/d/nvme_cache/vllm_kv, 194GB free) instead of being lost. The JSON has NO
# spaces deliberately -- EXTRA_ARGS is expanded unquoted in start_qwen.sh,
# so any space would split it into multiple argv words and corrupt the
# JSON; verified this survives word-splitting intact before using it here.
# kv_role=kv_both is required by vLLM's KVTransferConfig validator whenever
# kv_connector is supplied explicitly (unlike --kv-offloading-size's default
# path, which sets kv_connector via attribute assignment after construction
# and so never triggers that validator) -- confirmed by direct testing.
