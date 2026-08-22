#!/bin/bash
# Fast (~4 min), reliable A/B repro for the mamba-align-stale-state-queue.patch
# offloading regression documented in docs/mamba-align-prefill-leak.md.
#
# Runs 2 sequential 88K-token conversations against a running server
# (conv A turn 1, then conv B turn 1 -- big enough that conv B evicts conv A's
# GPU-local prefix cache -- then conv A turn 2, which should be served from
# the RAM/NVMe offload tier if offloading is working).
#
# Usage: bash micke-start.sh (or start whatever server config you want to
# test), wait for it to come up, then: bash test_offload_regression.sh
#
# PASS  (offloading works):    conv A turn 2 finishes in ~5s, qwen.log has
#                               a "hit N offloaded tokens after 0 GPU hit
#                               tokens" line with N close to the full prompt.
# FAIL  (offloading broken):   conv A turn 2 takes ~100s (a full solo-prompt
#                               reprocess), no "offloaded tokens" hit line.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="${TMPDIR:-/tmp}/offload_regression_test.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

mkfifo "$SCRATCH/fifo_ca" "$SCRATCH/fifo_cb"
exec 3<>"$SCRATCH/fifo_ca"
exec 4<>"$SCRATCH/fifo_cb"

venv/bin/python local_chat_test.py -p 4000 < "$SCRATCH/fifo_ca" > "$SCRATCH/ca.out" 2>&1 &
sleep 5   # force a different embedded timestamp than conv B (see below)
venv/bin/python local_chat_test.py -p 4000 < "$SCRATCH/fifo_cb" > "$SCRATCH/cb.out" 2>&1 &
sleep 2

wait_for() {
  local file=$1 count=$2
  while [ "$(grep -c '\[timing\]' "$file" 2>/dev/null)" -lt "$count" ]; do sleep 2; done
}

echo "Summarize the reference material above in 2 sentences." >&3
wait_for "$SCRATCH/ca.out" 1
echo "conv A turn1 done: $(date +%T)"

echo "Summarize the reference material above in 2 sentences." >&4
wait_for "$SCRATCH/cb.out" 1
echo "conv B turn1 done: $(date +%T)"

echo "Now restate that in one sentence." >&3
wait_for "$SCRATCH/ca.out" 2
echo "conv A turn2 done: $(date +%T)"

exec 3>&- 4>&-

echo
echo "=== conv A turn 2 timing (this is the one that matters) ==="
tail -1 "$SCRATCH/ca.out" | grep -o '\[timing\].*'
echo
echo "=== qwen.log: did the offload tier get hit? ==="
grep "offloaded tokens after" qwen.log | tail -3 || echo "(no hits found -- offloading did not serve this request)"
