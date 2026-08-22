#!/bin/sh
# Cap the fs KV-offload cache before starting -- vLLM's FileSystemTierManager
# has no size limit or eviction of its own (see prune_kv_offload_cache.sh's
# header and docs/mamba-align-prefill-leak.md's "no disk cap" TODO). This
# only catches growth between server restarts; for a long-running session,
# add a cron entry too, e.g.:
#   */30 * * * * bash $(pwd)/prune_kv_offload_cache.sh
bash "$(dirname "$0")/prune_kv_offload_cache.sh" /d/nvme_cache/vllm_kv || true
ITERATION_LOG=1 \
REQUEST_LOG_DIR=$(pwd)/requests \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_CACHE_METRICS_WINDOW=3 \
PREFIX_CACHE=1 \
CTX=long \
MAX_LEN=140000 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":21474836480,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv"}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
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
