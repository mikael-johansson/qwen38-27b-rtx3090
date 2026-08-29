#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Same base config as batch mode, plus MTP speculative decoding: the checkpoint
# keeps Qwen's multi-token-prediction head, so the model drafts 3-4 tokens ahead
# and verifies them in one pass. Measured on realistic chat prompts with the
# `-fast` model variant (see "Fast variant" below): ~114 tok/s at the model's
# default sampling, ~124 tok/s greedy (vs 46 tok/s without speculation).
# What makes 4 drafts pay off, in order of importance:
#  - the drafter scores a 40k-token draft head (build_draft_vocab.py) — and the
#    id list matters: a vocabulary counted over the model's OWN outputs covers
#    97.5% of what it generates (96% on code); the earlier web-text list only 92%
#    (83% on code), and every miss is a forced rejection (108 vs 98 tok/s greedy)
#  - the MTP module and lm_head requantized to int4 with GPTQ calibrated on the
#    model's hidden states (drafter/): 850 -> 215 MB per draft, 1.27 -> 0.65 GB
#    lm_head per verify, +0.6% perplexity, acceptance unchanged
#  - patches/spec-decode-attn.patch: split-KV attention for the 5-query verify
#    step (FA2 leaves 58 of 82 SMs idle there); patches/sampler-...: sort-free
#    top-k, multi-block softmax, drafts truncated to the target's top-k/top-p
#  - draft_sample_method=probabilistic: drafts are sampled, not argmax'ed, which
#    lifts acceptance at temperature > 0
# Speculative decoding is exact: none of this changes what gets sampled.
#
# CTX=fast (default here): FlashAttention + bf16 KV, 4 drafts, 64k context.
# CTX=long: fp8 KV via FlashInfer, 150k context, 3 drafts (k=4 crashes on
#   FlashInfer as soon as one request finishes while another is mid-generation,
#   vLLM 0.27.1); the split-KV attention patch is bf16-KV only, so ~90/98 tok/s.
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/), 200k context with MTP.
#
# Fast variant: MODEL defaults to models/Qwen3.8-27B-W4A16-AutoRound-fast when it
# exists (int4-GPTQ lm_head + MTP, own-output draft vocab; drafter/README.md), else
# the base dir (int8 lm_head/MTP: ~108/107 tok/s with the shipped draft vocab).
#
# max-num-seqs is 8 here: fewer state slots to reserve (each request holds
# k+1 recurrent-state slots), and past a handful of concurrent users you
# should be running batch mode anyway. Int8 activations are pointless at
# batch size 1 (memory-bound), so this mode stays W4A16.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "$MODEL" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-}
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (docs/gotchas.md, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=long (default): fp8 KV via FlashInfer, 150k context, 3 drafts.
# CTX=fast: bf16 KV via FlashAttention, ~64k context, 4 drafts (~+7%).
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/ in this repo, run kvarn/install.sh
#           once), 200k context with MTP, ~5% slower (see docs/long-context.md).
CTX=${CTX:-fast}
# SPEC=mtp (default): Qwen's own MTP head, k drafts chained (the numbers above).
# SPEC=dflash2: the DFlash2 block drafter (incoai/Qwen3.8-27B-DFlash2, requantized
#   to W4A16 by this repo: fetch_dflash2.py), 7 drafts in ONE non-autoregressive
#   pass + a path selector; runs on vLLM's V2 model runner
#   (patches/dflash2-backport.patch). CTX=fast only (bf16 KV / FLASH_ATTN; the
#   drafter's block attention is non-causal); see README "DFlash2".
SPEC=${SPEC:-mtp}
# SPEC_ATTN=1: split-KV Triton attention for the multi-query verify step
# (patches/spec-decode-attn.patch); bf16 KV only, so CTX=fast only.
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  ATTN_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$CTX" = "huge" ]; then
  MAX_LEN=${MAX_LEN:-200000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
else
  MAX_LEN=${MAX_LEN:-150000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype fp8"
fi
if [ "$SPEC" = "dflash2" ]; then
  if [ "$CTX" != "fast" ]; then
    echo "SPEC=dflash2 is CTX=fast only (bf16 KV, FLASH_ATTN); CTX=$CTX keeps SPEC=mtp" >&2
    SPEC=mtp
  fi
fi
if [ "$SPEC" = "dflash2" ]; then
  if [ -z "$DRAFT" ]; then
    for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
      [ -f "$REPO/models/$d/model.safetensors" ] && DRAFT=$REPO/models/$d && break
    done
  fi
  [ -n "$DRAFT" ] || { echo "SPEC=dflash2 needs the drafter: venv/bin/python fetch_dflash2.py" >&2; exit 1; }
  # Lookup-augmented drafting: when the model is reproducing something from its context,
  # draft from the context instead of from the drafter
  # (patches/dflash2-lookup-drafting.patch).
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # DFLASH_TOKENS is the *verify* block, which no longer has to equal the drafter's: the
  # DFlash2 checkpoint always proposes the 7 tokens it was trained for, and any position
  # past that is filled from the request's own context, costing the drafter nothing. The
  # block length is adaptive -- the long block is only scheduled while the lookup is
  # actually firing -- so ordinary steps still verify 8 tokens.
  #
  # DFLASH_TOKENS=15 is "reproduction mode": +50% where the model reproduces its context
  # (388 vs 259 tok/s reproducing a document verbatim) and +9% on the short-prompt C1 set,
  # against 3-20% on long-context work that mixes prose with quoting, 4 request slots
  # instead of 8 and 56k of context instead of 64k. Worth setting for a coding assistant
  # applying edits or a RAG front-end quoting sources; the default stays 7.
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
  # The split-KV verify attention (patches/spec-decode-attn.patch) sizes its partial
  # buffers once for the longest query block it will see -- a captured CUDA graph holds
  # their addresses, so they must not be grown later.
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    # Adaptive block length means the worker tells the scheduler how many draft tokens to
    # put up for verification next step, and vLLM only feeds that back on the synchronous
    # scheduling path (async scheduling pads every decode step to num_speculative_tokens).
    # Measured cost of losing async scheduling at batch 1: under 1%.
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # Memory: patches/hybrid-kv-groups-v2-cudagraph.patch stops the drafter's 5
  # sliding-window layers from padding the target's attention/GDN layers (78 instead of
  # 105 KB of pool per token), which is what makes 64k reachable here. The V2 runner's
  # profiled activation peak swings ~1 GiB between starts, so the pool is pinned by bytes
  # rather than by gpu-memory-utilization: 5.2 GiB -> 69,758 tokens = 1.06x at 64k,
  # leaving ~1.1 GiB for transients (the same margin MTP mode runs with). Soak-tested
  # with a 60k prompt, 4x16k concurrent and 8x4k generations. Lower it if you also run
  # something else on the card; KV_MEM= (empty) falls back to GPU_UTIL.
  #
  # A longer verify block costs pool twice: bigger CUDA graphs, and one aligned recurrent
  # state page per request per speculative block. MAX_SEQS is what that scales with, so
  # single-user mode keeps 4 slots when the block is long.
  if [ "$DRAFT_TOKENS" -gt 7 ]; then
    # 4 slots and 56k instead of 8 and 64k: the aligned state pages and the bigger decode
    # graphs are what the long block costs, and this is where they still fit next to the
    # 5.2 GiB pool (57,669 tokens). DFLASH_TOKENS=7 gets 8 slots and 64k back.
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-57344}
    KV_MEM=${KV_MEM-5583457484}
    # Decode graphs are captured for both block lengths (the drafter's and the full verify
    # block), or the short step -- the common one -- runs piecewise and costs 8%. That is
    # 1.8 GiB of graphs instead of 1.45.
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
  else
    MAX_LEN=${DFLASH_MAX_LEN:-65536}
    KV_MEM=${KV_MEM-5583457484}
    # If you tune GPU_UTIL instead, make the V2 runner count its CUDA graphs (~1.2-1.3 GiB
    # at these capture sizes) as well:
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  # The V2 model runner captures decode graphs in multiples of k+1 tokens: cover MAX_SEQS requests.
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
else
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
fi

# Pin the KV pool in bytes, for BOTH spec paths (this used to be dflash2-only).
# gotcha 18: the profiled activation peak varies by ~1 GiB between starts of the
# same config, so sizing the pool from GPU_UTIL is not reproducible. Pinning
# bytes is. It also makes co-tenancy arithmetic exact -- measured on this box:
#
#   total vLLM footprint = 16330 MiB (weights + graphs + activations, constant)
#                        + KV_MEM
#
# so with something else resident (e.g. whisper.cpp STT at ~700 MiB):
#
#   KV_MEM_MiB = 24576 - 16330 - <other resident> - <spike> - <margin>
#
# The DeltaNet/GDN prefill spike measured 1208 MiB at 2 concurrent requests with
# a 40k prefill. 2026-08-27.
[ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"

# PREFIX_CACHE=1: reuse the KV of a shared prompt prefix across requests, and resume the
# recurrent (GDN) state from the last cached block boundary instead of re-running the prompt.
# Turn-2+ of a chat with a 24k document goes from ~23 s to ~1 s; costs one extra state page
# per request (~16% of the KV pool). Hybrid models keep this opt-in upstream.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# ITERATION_LOG=1: log one line per engine step to stdout/qwen.log, naming
# the actual requests being prefilled/decoded and their real throughput —
# "what's eating GPU cycles right now", useful when requests overlap. Reuses
# vLLM's own --enable-logging-iteration-details (patches/iteration-details-
# reqids.patch adds the request-id + corrected-timing formatting on top).
# High volume while decoding (a line per step, not just per batch) — see
# REQUEST_LOGGING_SPEC.md Part 5 "Expected volume" before leaving this on
# routinely.
if [ "${ITERATION_LOG:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-logging-iteration-details ${EXTRA_ARGS}"
fi

# ASYNC_SCHED=0 (set above for a long DFlash2 verify block) runs the scheduler
# synchronously, which is the only path on which vLLM lets the worker choose how many draft
# tokens to put up for verification. Note --async-scheduling is already the default in
# 0.27.1: --no-async-scheduling is what turns it off.
ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)

# --kv-offloading-size: vLLM's CPU-offload connector creates a multi-GB
# mmap file at /dev/shm/vllm_offload_<engine_id>.mmap. patches/offload-
# cleanup-on-init-failure.patch makes it clean up after itself when its own
# startup fails, but that's an in-process fix — it structurally cannot help
# against SIGKILL, an OOM-kill, or a hard crash that skips Python entirely.
# This is the unconditional backstop: only touches /dev/shm if no vllm
# server is currently running (so it can never delete a live instance's
# file), which is also exactly the situation that left one behind.
if ! pgrep -f "$REPO/venv/bin/vllm serve" > /dev/null 2>&1; then
  stale_offload_files=$(ls /dev/shm/vllm_offload_*.mmap 2>/dev/null)
  if [ -n "$stale_offload_files" ]; then
    echo "start_qwen.sh: removing stale /dev/shm offload file(s) from a previous run (no vllm server currently running):" >&2
    echo "$stale_offload_files" >&2
    rm -f /dev/shm/vllm_offload_*.mmap
  fi
fi

export PATH="$REPO/venv/bin:$PATH"
# Overridable: expandable_segments needs CUDA VMM, which WSL2's paravirt
# driver rejects ("CUDA driver error: device not ready" during Marlin repack)
# — set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False in .env on WSL2.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

# patches/request-logging.patch imports request_logging.py from the repo
# root (it's not vendored into vllm/) — always needed, not just when
# REQUEST_LOG_DIR is set, since the patch imports it unconditionally at
# module load time (degrades to a silent no-op without this, doesn't crash).
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# REQUEST_LOG_DIR: unset (default) = per-request disk logging off. Set to a
# directory (e.g. REQUEST_LOG_DIR=$REPO/qwen.log.d/requests) to enable full
# request/response logging — see REQUEST_LOGGING_SPEC.md.
if [ -n "$REQUEST_LOG_DIR" ]; then
  export REQUEST_LOG_DIR
fi

# VISION=1 loads the model's vision tower. Default stays text-only:
# --language-model-only sets limit_mm_per_prompt=0 for every modality, which
# makes vLLM's _mark_tower_model skip *instantiating* the tower, so it costs no
# VRAM at all.
#
# VISION_OFFLOAD selects where the ViT's 27 transformer blocks (0.7664 GiB of
# the tower's 0.8582, BF16 — the ViT is excluded from the W4A16 quant) live,
# since that is KV pool you are giving up. Stock --cpu-offload-gb cannot reach a
# vision tower at all; see patches/vision-tower-cpu-offload.patch for why and
# for the measured numbers.
#
#   prefetch  (default) streamed H2D into a small staging pool per block,
#             overlapped with the previous block's compute. +0.72 GiB of KV
#             pool back; free (within noise) on images >= 896px, ~+45 ms on
#             thumbnails.
#   uva       zero-copy reads over PCIe inside every GEMM. +0.78 GiB — a little
#             more, since there is no staging buffer — but 2.8-4.8x on image
#             latency.
#   off       weights in VRAM. NB: at MAX_LEN=140000 the engine then refuses to
#             start at all (needs 5.34 GiB of KV pool, has 5.18).
#
# With VISION=1 the caller must also pass, via EXTRA_ARGS, at minimum a
# --limit-mm-per-prompt: the 999-per-modality default makes startup memory
# profiling reserve an absurd activation peak. See micke-start-vision.sh.
LM_ONLY_ARG="--language-model-only"
if [ "${VISION:-0}" = "1" ]; then
  LM_ONLY_ARG=""
  # Fail loudly if the patch is not installed. Without it vLLM does not know
  # VLLM_VISION_OFFLOAD_BACKEND, silently ignores it, and loads the tower fully
  # resident — which at MAX_LEN=140000 means the engine refuses to start, with
  # an error about KV cache size that gives no hint the vision setting was
  # dropped.
  if [ "${VISION_OFFLOAD:-prefetch}" != "off" ]; then
    _sp=$(venv/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))' 2>/dev/null)
    if ! grep -q VLLM_VISION_OFFLOAD_BACKEND "$_sp/envs.py" 2>/dev/null; then
      echo "VISION_OFFLOAD=${VISION_OFFLOAD:-prefetch} needs patches/vision-tower-cpu-offload.patch," >&2
      echo "which is not applied to $_sp. Apply it with:" >&2
      echo "  patch -p1 -d \"$_sp\" < patches/vision-tower-cpu-offload.patch" >&2
      exit 1
    fi
  fi
  case "${VISION_OFFLOAD:-prefetch}" in
    prefetch)
      export VLLM_VISION_OFFLOAD_BACKEND=prefetch
      export VLLM_VISION_PREFETCH_STEP=${VISION_PREFETCH_STEP:-2} ;;
    uva)
      export VLLM_VISION_OFFLOAD_BACKEND=uva
      export VLLM_VISION_CPU_OFFLOAD_GB=${VISION_OFFLOAD_GB:-1} ;;
    off)
      export VLLM_VISION_OFFLOAD_BACKEND=off ;;
    *)
      echo "VISION_OFFLOAD must be prefetch, uva or off (got '$VISION_OFFLOAD')" >&2
      exit 1 ;;
  esac
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${LM_ONLY_ARG} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
# SPEC=none disables speculative decoding entirely (diagnostic: the MTP drafter's
# weight-unpack transient dominates the startup peak). Anything else keeps MTP.
if [ "$SPEC" = "none" ]; then SPEC_ARG=""; else SPEC_ARG="--speculative-config $SPEC_CFG"; fi
  --max-num-batched-tokens ${MNBT:-2048} \
  ${SPEC_ARG} \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  ${EXTRA_ARGS}
