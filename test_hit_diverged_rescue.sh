#!/bin/bash
# Repro/verification for patches/hit-diverged-boundary-rescue.patch.
#
# Provokes the HIT_DIVERGED pattern the rescue targets: conversation turn 1
# (long prompt), short reply, then turn 2 while the GPU pool still holds the
# full-attention chain (no eviction pressure in between -- the point is a deep
# LOCAL attention hit whose mamba-group local hit is shallower, with the
# offload tier able to back the deep boundary).
#
# PASS:  turn 2 logs either "HIT_DIVERGED RESCUED" or no divergence at all
#        (both mean: no reprefill), AND turn 2 total time is seconds, never a
#        full reprefill (~100s at this size).
# FAIL:  any "HIT_DIVERGED RECONCILE" during the run, or a slow turn 2.
#
# Usage: server up (bash micke-start.sh), then: bash test_hit_diverged_rescue.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="${TMPDIR:-/tmp}/hit_diverged_rescue_test.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

LOG_START=$(wc -l < qwen.log)

mkfifo "$SCRATCH/fifo"
exec 3<>"$SCRATCH/fifo"
venv/bin/python local_chat_test.py -p 2000 < "$SCRATCH/fifo" > "$SCRATCH/conv.out" 2>&1 &

wait_for() {
  local count=$1
  while [ "$(grep -c '\[timing\]' "$SCRATCH/conv.out" 2>/dev/null)" -lt "$count" ]; do sleep 2; done
}

sleep 2
echo "Summarize the reference material above in 2 sentences." >&3
wait_for 1
echo "turn 1 done: $(date +%T)"

# Let the offload stores for turn 1 (prefill + decode tail) settle.
sleep 8

T2_START=$(date +%s)
echo "Now restate that in one sentence." >&3
wait_for 2
T2_SECS=$(( $(date +%s) - T2_START ))
echo "turn 2 done: $(date +%T) (${T2_SECS}s)"
exec 3>&-

echo
echo "=== turn 2 timing ==="
tail -1 "$SCRATCH/conv.out" | grep -o '\[timing\].*' || true

NEWLOG="$SCRATCH/qwen_tail.log"
tail -n "+$((LOG_START + 1))" qwen.log > "$NEWLOG"
RESCUED=$(grep -c "HIT_DIVERGED RESCUED" "$NEWLOG" || true)
RECONCILE=$(grep -c "HIT_DIVERGED RECONCILE" "$NEWLOG" || true)
echo "=== qwen.log since test start: RESCUED=$RESCUED RECONCILE=$RECONCILE ==="
grep "HIT_DIVERGED" "$NEWLOG" | tail -5 || true

FAIL=0
if [ "$RECONCILE" -gt 0 ]; then
  echo "FAIL: $RECONCILE HIT_DIVERGED RECONCILE event(s) -- deep hit was discarded"
  FAIL=1
fi
if [ "$T2_SECS" -gt 30 ]; then
  echo "FAIL: turn 2 took ${T2_SECS}s -- looks like a reprefill"
  FAIL=1
fi
if [ "$FAIL" = 0 ]; then
  if [ "$RESCUED" -gt 0 ]; then
    echo "PASS: divergence rescued ($RESCUED event(s)), turn 2 in ${T2_SECS}s"
  else
    echo "PASS: no divergence at all, turn 2 in ${T2_SECS}s"
    echo "(solo runs often don't diverge; for a stronger check run"
    echo " venv/bin/python simulate_hermes_traffic.py --threads 2 and assert"
    echo " zero RECONCILE with nonzero RESCUED over the run)"
  fi
fi
exit $FAIL
