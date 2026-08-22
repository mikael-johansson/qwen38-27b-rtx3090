#!/bin/bash
# SUPERSEDED for normal use by patches/fs-tier-disk-cap.patch, which adds
# continuous, in-process LRU disk capping directly to vLLM's
# FileSystemTierManager (a background thread inside the server itself,
# checked every couple of minutes for the life of the process -- see
# micke-start.sh's "max_disk_gib":200). This external, launch-time-only
# script can't keep up with a long-running session by itself (it only ever
# ran once, at server startup) and is kept here only as a manual/one-off
# cleanup tool -- e.g. to reclaim disk space right now without restarting
# the server, or for a launch config that doesn't apply the patch above.
#
# Cap the KV-offloading fs secondary tier's disk usage, LRU-ish (evicts the
# least-recently-*accessed* .bin files first, using atime -- confirmed
# `relatime` is active on the cache filesystem here, so atime is a real,
# if slightly coarse, recency signal, not a no-op).
#
# Why this exists: vLLM 0.27.1's FileSystemTierManager (v1/kv_offload/
# tiering/fs/manager.py) and FileMapper (v1/kv_offload/file_mapper.py) have
# NO size cap or eviction policy at all -- confirmed by reading both
# classes' full __init__ signatures and the SecondaryTierFactory
# construction path (any extra kv_connector_extra_config.secondary_tiers[]
# key besides "type" is passed straight through as a kwarg; there is
# nothing to catch a size/quota option). Unlike the primary CPU tier
# (cpu_bytes_to_use, LRU/ARC eviction built in), the fs tier just writes
# files under root_dir forever. See docs/mamba-align-prefill-leak.md's
# "Separate TODO" section for the investigation that found this (it filled
# a 234G disk to 100% over a few hours of testing).
#
# Usage: bash prune_kv_offload_cache.sh [dir] [max_bytes] [target_bytes]
#   dir          default: /d/nvme_cache/vllm_kv
#   max_bytes    default: 200GB -- prune only triggers above this
#   target_bytes default: 180GB -- prunes down to here once triggered, so
#                it doesn't need to run again immediately
#
# Safe to run against a live server: deleting a block file that's mid-read
# just makes that one lookup a MISS, which the offloading connector already
# handles gracefully (falls back to recomputing that chunk, same as any
# ordinary cache miss) -- confirmed by the whole investigation in
# docs/mamba-align-prefill-leak.md, which is full of exactly this kind of
# miss happening for unrelated reasons without incident.
set -euo pipefail

DIR="${1:-/d/nvme_cache/vllm_kv}"
MAX_BYTES="${2:-$((200 * 1024 * 1024 * 1024))}"
TARGET_BYTES="${3:-$((180 * 1024 * 1024 * 1024))}"

if [ ! -d "$DIR" ]; then
  echo "prune_kv_offload_cache: $DIR does not exist, nothing to do"
  exit 0
fi

CUR=$(du -sb "$DIR" 2>/dev/null | awk '{print $1}')
CUR=${CUR:-0}

if [ "$CUR" -le "$MAX_BYTES" ]; then
  echo "prune_kv_offload_cache: $DIR is $((CUR / 1024 / 1024 / 1024))G, under the $((MAX_BYTES / 1024 / 1024 / 1024))G cap -- nothing to do"
  exit 0
fi

echo "prune_kv_offload_cache: $DIR is $((CUR / 1024 / 1024 / 1024))G, over the $((MAX_BYTES / 1024 / 1024 / 1024))G cap -- evicting oldest-accessed .bin files down to $((TARGET_BYTES / 1024 / 1024 / 1024))G"

freed=0
deleted=0
# Oldest atime first. Only *.bin (actual KV block data) -- never touches
# the tiny per-run config.json FileMapper writes alongside them.
while IFS=' ' read -r atime size path; do
  [ "$((CUR - freed))" -le "$TARGET_BYTES" ] && break
  if rm -f -- "$path" 2>/dev/null; then
    freed=$((freed + size))
    deleted=$((deleted + 1))
  fi
done < <(find "$DIR" -type f -name '*.bin' -printf '%A@ %s %p\n' 2>/dev/null | sort -n)

echo "prune_kv_offload_cache: deleted $deleted files, freed $((freed / 1024 / 1024 / 1024))G -- $DIR now approximately $(( (CUR - freed) / 1024 / 1024 / 1024 ))G"
