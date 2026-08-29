#!/bin/sh
# Preflight: vLLM's request_memory() gate refuses to start unless
# free_vram >= GPU_UTIL x total. At GPU_UTIL=0.97 that is 22.85 of 23.56 GiB, so
# nothing else may hold the card at startup -- whisper's ~700 MiB is enough to
# fail it, with a vLLM error that does not mention whisper. Say so plainly here.
_util=${GPU_UTIL:-0.97}
_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
_tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$_free" ] && [ -n "$_tot" ]; then
  # vLLM measures against torch's visible total (23.56 GiB = 24125 MiB), which is
  # ~451 MiB below nvidia-smi's 24576. Match it, or this gate is stricter than the
  # real one and refuses starts that would have worked.
  _need=$(awk -v u="$_util" -v t="$_tot" 'BEGIN{printf "%d", u*(t-451)}')
  if [ "$_free" -lt "$_need" ]; then
    echo "start: GPU_UTIL=$_util needs ${_need} MiB free, only ${_free} MiB is." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  holding: /' >&2
    echo "" >&2
    echo "  If that is whisper-server, stop it, start vLLM, then bring it back:" >&2
    echo "    systemctl --user stop whisper-server" >&2
    echo "    sh micke-start-vision.sh" >&2
    echo "    systemctl --user start whisper-server     # once vLLM is serving" >&2
    echo "  (whisper-cpp/hermes/start-stack.sh does all three.)" >&2
    echo "" >&2
    echo "  Or run with a lower utilisation, which leaves room for it:" >&2
    echo "    GPU_UTIL=0.95 sh micke-start-vision.sh    # 173,820 tokens instead of 188,764" >&2
    exit 1
  fi
fi

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
KV_MEM=${KV_MEM:-6452936704} \
GPU_UTIL=${GPU_UTIL:-0.97} \
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
# GPU_UTIL=0.97 + KV_MEM=6452936704 -> 188,764 KV tokens (1.35x at MAX_LEN=140000).
#
# GPU_UTIL is NOT inert when KV_MEM is pinned. vLLM logs "skipped memory
# profiling" and does not use KV_MEM as the pool size -- it takes
# min(KV_MEM, what fits in util x total after weights+graphs). So GPU_UTIL is
# what actually sets the pool, and KV_MEM only caps it. Measured ladder, each
# step soak-tested with 4 concurrent 39-49k prompts generating 12288 tokens
# each WITH whisper serving throughout:
#
#   util   tokens    runtime peak   headroom   result
#   0.93   158,089   -              -          (previous default)
#   0.95   173,820   22510          2066       4/4 ok, 0 whisper fail
#   0.96   180,898   22730          1846       4/4 ok, 0 whisper fail
#   0.97   188,764   23010          1566       4/4 ok, 0 whisper fail   <- default
#   0.98   195,842   23190          1386       4/4 ok, 0 whisper fail
#
# 0.97 is chosen over 0.98 for margin, not because 0.98 failed: the largest
# runtime rise ever observed is 1237 MiB, so 0.97 leaves ~330 MiB beyond it and
# 0.98 only ~150. Set GPU_UTIL=0.98 if you want the extra 7,078 tokens.
#
# CAVEAT: a ~25 min soak cannot prove long-term stability. docs/gotchas.md #4
# warns that this path "survives short benchmarks, which is exactly how it fools
# you", and the 08-28 illegal-memory-access crash took 38 h to appear. These
# settings are past the 0.93 that gotcha soak-tested. Revert with GPU_UTIL=0.93.
#
# START ORDER: still stop whisper-server before starting vLLM. The startup peak
# is a ~3 s transient during the MTP drafter's embed/lm_head unpack and it varies
# 23113-23853 MiB between runs, so the margin against whisper's 724 MiB plateau
# is not reliable. hermes/start-stack.sh sequences it.
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
