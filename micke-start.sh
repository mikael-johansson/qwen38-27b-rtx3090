#!/bin/sh
ITERATION_LOG=0 \
REQUEST_LOG_DIR=$(pwd)/requests \
VLLM_LOG_STATS_INTERVAL=10 \
VLLM_CACHE_METRICS_WINDOW=3 \
PREFIX_CACHE=1 \
CTX=long \
MAX_LEN=140000 \
PYTHONHASHSEED=0 \
VLLM_OFFLOAD_EAGLE_FALLBACK=0 \
VLLM_LOGGING_CONFIG_PATH=$(pwd)/single-user/logging-to-file-debug.json \
EXTRA_ARGS='  --enable-auto-tool-choice --tool-call-parser qwen3_xml --long-prefill-token-threshold 832 --kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":21474836480,"secondary_tiers":[{"type":"fs","root_dir":"/d/nvme_cache/vllm_kv","max_disk_gib":200}]}} --enable-cumem-allocator' \
bash single-user/start_qwen.sh
# ITERATION_LOG=0 (2026-08-24, was 1) -- --enable-logging-iteration-details
# logs one line per non-dummy engine step ("generation [...] | GPU KV cache
# usage: ...%"); under MTP spec-decode that's nearly every decode step, so
# in practice it fired on almost every step of every live generation --
# too frequent to read live. Left off by default now that
# v1/engine/request_lifecycle_log.py (NEW REQUEST / FINISHED lines, one
# per request instead of one per step) covers what this was actually being
# read for. Still available -- set ITERATION_LOG=1 to turn the per-step
# detail back on for debugging.
#
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
# --long-prefill-token-threshold 512 -- caps how many tokens of a single
# request's prefill get scheduled per step, even once that request is
# already admitted into the running batch. Without this (default 0 =
# unlimited), an admitted long prompt is entitled to the entire
# --max-num-batched-tokens (2048) budget every single step until its own
# prefill finishes, starving other running/waiting requests of that step's
# budget the whole time -- confirmed by reading scheduler.py's schedule():
# the running-phase loop runs before the waiting-phase loop and hands out
# token_budget FCFS with no per-request cap unless this is set. At 512,
# a big prefill now yields the rest of the budget back every step, letting
# other conversations' decode and other prefills' chunks interleave instead
# of one long prompt monopolizing the server until it's done. See
# 2026-08-22 conversation.
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
