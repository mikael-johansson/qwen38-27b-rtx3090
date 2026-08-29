#!/bin/bash
# Start vLLM and whisper.cpp together, in the only order that works, and VERIFY
# that vLLM actually received the full argument set before declaring success.
#
#   ./start-with-whisper.sh              # normal start
#   GPU_UTIL=0.95 ./start-with-whisper.sh
#   ./start-with-whisper.sh --no-whisper # vLLM only
#
# Why each step exists:
#  1. whisper must be stopped first. vLLM's request_memory() gate needs
#     GPU_UTIL x total free at startup, and whisper's ~700 MiB breaks it.
#  2. The card must be genuinely idle AND stable, not merely under a threshold
#     (docs/gotchas.md #2: profiling a still-draining GPU silently undersizes
#     the KV pool by ~25% with no warning).
#  3. A killed vLLM leaves a 20 GiB /dev/shm segment; the next start then dies
#     with "OSError: [Errno 14] Bad address" that names neither shm nor memory.
#  4. ARG VERIFICATION. A malformed continuation inside the exec silently drops
#     every flag after it -- the server comes up healthy with speculative
#     decoding, prefix caching and KV offload missing. `sh -n` does not catch it.
#     This cost a full day of invalid measurements. Never skip this check.
set -uo pipefail
cd "$(dirname "$0")"
HEALTH=http://127.0.0.1:18020/health
LOG=${LOG:-/tmp/vllm-start.log}
WANT_WHISPER=1
[ "${1:-}" = "--no-whisper" ] && WANT_WHISPER=0

# Flags that MUST appear in vLLM's "non-default args" line. If any is missing the
# command line was truncated and every measurement taken from it is worthless.
REQUIRED="speculative_config enable_prefix_caching kv_transfer_config \
enable_auto_tool_choice tool_call_parser reasoning_parser enable_cumem_allocator \
limit_mm_per_prompt max_num_batched_tokens"

say() { printf '%s\n' "$*"; }

say "1/6 stopping whisper-server"
systemctl --user stop whisper-server 2>/dev/null

say "2/6 stopping vLLM"
for p in $(ps -eo pid,cmd --no-headers | awk '$0 ~ /VLLM::EngineCore|vllm serve/ && $0 !~ /awk/{print $1}'); do kill -TERM "$p" 2>/dev/null; done
sleep 8
for p in $(ps -eo pid,cmd --no-headers | awk '$0 ~ /VLLM::EngineCore|vllm serve/ && $0 !~ /awk/{print $1}'); do kill -KILL "$p" 2>/dev/null; done

say "3/6 waiting for the GPU to drain AND hold steady (gotcha 2)"
prev=-1; stable=0; u=0
for _ in $(seq 1 150); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$u" -lt 60 ] && [ "$u" = "$prev" ]; then stable=$((stable+1)); else stable=0; fi
  [ "$stable" -ge 5 ] && break
  prev=$u; sleep 2
done
[ "$stable" -ge 5 ] || { say "    GPU never settled (still ${u} MiB) -- something else is holding it:"; \
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/      /'; exit 1; }
say "    idle at ${u} MiB"

say "4/6 clearing orphaned /dev/shm offload segments"
for f in /dev/shm/vllm_offload_*.mmap; do
  [ -e "$f" ] || continue
  if fuser "$f" >/dev/null 2>&1; then say "    in use, keeping: $f"
  else say "    removing orphan: $f"; rm -f "$f"; fi
done

say "5/6 starting vLLM"
setsid nohup sh micke-start-vision.sh > "$LOG" 2>&1 </dev/null &
for _ in $(seq 1 150); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $HEALTH 2>/dev/null)" = "200" ] && break
  if grep -q "EngineCore failed" "$LOG" 2>/dev/null; then
    say "    FAILED. Root cause:"
    grep -hoE "ValueError: .{0,200}|torch.OutOfMemoryError: .{0,120}|OSError: .{0,120}" "$LOG" | tail -1 | sed 's/^/      /'
    exit 1
  fi
  sleep 6
done
[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $HEALTH 2>/dev/null)" = "200" ] || { say "    never became healthy; see $LOG"; exit 1; }

# --- the check that would have caught the day I lost ---
ARGS=$(grep -m1 -oE "non-default args: \{.*" "$LOG" | sed 's/\x1b\[[0-9;]*m//g')
MISSING=""
for k in $REQUIRED; do
  printf '%s' "$ARGS" | grep -q "'$k'" || MISSING="$MISSING $k"
done
if [ -n "$MISSING" ]; then
  say "    ARGUMENT CHECK FAILED -- vLLM is running with flags missing:$MISSING"
  say "    The command line in single-user/start_qwen.sh is truncated."
  say "    Do not trust anything measured from this server. Fix and restart."
  exit 1
fi
say "    arg check passed (all $(printf '%s' "$REQUIRED" | wc -w) required flags present)"
grep -hoE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$LOG" | tail -2 | sed 's/^/    /'

if [ "$WANT_WHISPER" = 1 ]; then
  say "6/6 starting whisper-server"
  systemctl --user start whisper-server
  sleep 12
  systemctl --user is-active --quiet whisper-server || { say "    whisper failed to start"; exit 1; }
else
  say "6/6 skipping whisper (--no-whisper)"
fi

say ""
say "ready:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'
say "  GPU total: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
