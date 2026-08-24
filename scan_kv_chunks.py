#!/usr/bin/env python3
"""Offline byte-scan of stored KV-offload fs-tier chunk files.

RESTORE_CORRUPTION_PLAN4.md Phase A: find maximal runs of the scribble
byte (VLLM_DEBUG_SCRIBBLE_KV pattern, filled via .fill_() over every byte
of every KV-cache tensor before any request ran -- see
patches/restore-corruption-scribble-debug.patch) that survive into a
*stored* chunk file. A surviving run means: the live compute path never
wrote that byte range before the store job captured it.

Layout (read from the fs tier's own config.json, not hardcoded): each
chunk file holds one KV-cache-group's set of layers for one 832-token
block, one canonical "layer slot" of LAYER_SLOT_BYTES each, uniformly
sized across all groups regardless of how many of that group's layers are
real (a group with fewer layers than the file's total slot count has
trailing, always-unused padding -- not a bug, just don't misreport it as
one).

Usage:
    venv/bin/python scan_kv_chunks.py <fs_tier_root_dir> [--byte 0x7b] [--min-run 32]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


def find_config(root_dir: str) -> str:
    candidates = glob.glob(os.path.join(root_dir, "*", "config.json"))
    if not candidates:
        raise FileNotFoundError(f"no config.json under {root_dir}")
    return candidates[0]


def find_runs(data: bytes, byte_val: int, min_run: int):
    """Yield (start, length) for maximal runs of byte_val >= min_run.
    Vectorized via re (C implementation) -- a manual Python byte loop over
    ~29MB x 280 files is far too slow."""
    pattern = re.compile(re.escape(bytes([byte_val])) + b"{%d,}" % min_run)
    return [(m.start(), m.end() - m.start()) for m in pattern.finditer(data)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir")
    ap.add_argument("--byte", default="0x7b")
    ap.add_argument("--min-run", type=int, default=32)
    ap.add_argument("--max-files-per-group", type=int, default=0,
                     help="0 = all files")
    args = ap.parse_args()

    byte_val = int(args.byte, 0)
    cfg_path = find_config(args.root_dir)
    cfg = json.load(open(cfg_path))
    groups = cfg["kv_cache_groups"]
    tokens_per_block = cfg.get("tokens_per_hash", 832)

    print(f"config: {cfg_path}")
    for i, g in enumerate(groups):
        print(f"  group {i}: {len(g['layer_names'])} layers, "
              f"last={g['layer_names'][-1]!r}")

    r0_dir = args.root_dir.rstrip("/") + "_r0"
    if not os.path.isdir(r0_dir):
        # maybe root_dir already points at the _r0 dir's parent-with-config;
        # try sibling
        base = os.path.dirname(cfg_path)
        r0_dir = base + "_r0"
    print(f"scanning: {r0_dir}")

    files = sorted(glob.glob(os.path.join(r0_dir, "*", "*", "*.bin")))
    print(f"total files: {len(files)}")

    by_group = defaultdict(list)
    for f in files:
        m = re.search(r"_g(\d+)/", f)
        if not m:
            continue
        by_group[int(m.group(1))].append(f)

    # Determine LAYER_SLOT_BYTES from the largest single-group file size /
    # the max layer count across groups that still evenly divides every
    # file's total size -- but simplest: file_size / num_slots must be an
    # integer; probe using the group with the most layers first (it's most
    # likely to occupy the file with least/no padding).
    sample_group = max(range(len(groups)), key=lambda i: len(groups[i]["layer_names"]))
    sample_file = by_group[sample_group][0]
    file_size = os.path.getsize(sample_file)
    n_layers_sample = len(groups[sample_group]["layer_names"])
    if file_size % n_layers_sample == 0:
        layer_slot_bytes = file_size // n_layers_sample
    else:
        raise RuntimeError(
            f"file_size={file_size} not divisible by sample group's "
            f"layer count={n_layers_sample}; layout assumption wrong"
        )
    print(f"file_size={file_size} layer_slot_bytes={layer_slot_bytes} "
          f"(from group {sample_group}, {n_layers_sample} layers)")

    results = {}  # group -> layer_idx -> list of (file, offset_in_layer, length)
    padding_hits = defaultdict(list)  # group -> list of (file, offset, length) beyond real layers

    for gi in sorted(by_group):
        n_layers = len(groups[gi]["layer_names"])
        real_bytes = n_layers * layer_slot_bytes
        flist = by_group[gi]
        if args.max_files_per_group:
            flist = flist[: args.max_files_per_group]
        layer_hits = defaultdict(list)
        for f in flist:
            data = open(f, "rb").read()
            if len(data) != file_size:
                print(f"  WARNING: {f} size {len(data)} != expected {file_size}")
            for start, length in find_runs(data, byte_val, args.min_run):
                if start < real_bytes:
                    layer_idx = start // layer_slot_bytes
                    off_in_layer = start - layer_idx * layer_slot_bytes
                    # a run might cross a layer boundary; just report by start layer
                    layer_hits[layer_idx].append((os.path.basename(f), off_in_layer, length))
                else:
                    padding_hits[gi].append((os.path.basename(f), start - real_bytes, length))
        results[gi] = layer_hits

    print("\n=== SURVIVING SCRIBBLE RUNS (real layer data -- these are the finding) ===")
    for gi in sorted(results):
        n_layers = len(groups[gi]["layer_names"])
        print(f"\ngroup {gi} ({n_layers} real layers, layer_slot_bytes={layer_slot_bytes}):")
        layer_hits = results[gi]
        if not layer_hits:
            print("  (no surviving runs in real layer data)")
            continue
        for layer_idx in sorted(layer_hits):
            hits = layer_hits[layer_idx]
            layer_name = groups[gi]["layer_names"][layer_idx]
            total_files = len(by_group[gi][: args.max_files_per_group] if args.max_files_per_group else by_group[gi])
            # group by (offset, length) SIGNATURE to see distinct patterns,
            # and count how many distinct files show each signature.
            by_sig = defaultdict(set)
            for fname, off, length in hits:
                by_sig[(off, length)].add(fname)
            n_files_hit = len(set(h[0] for h in hits))
            print(f"  layer {layer_idx} ({layer_name}): {n_files_hit}/{total_files} files affected")
            for (off, length), fnames in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
                sample_names = sorted(fnames)[:3]
                print(f"      offset={off:>8} length={length:>8}  "
                      f"in {len(fnames)}/{total_files} files  e.g. {sample_names}")

    print("\n=== TRAILING PADDING (beyond real layer count -- expected, not a bug) ===")
    for gi in sorted(padding_hits):
        n_layers = len(groups[gi]["layer_names"])
        pad_bytes = file_size - n_layers * layer_slot_bytes
        n_files = len(set(h[0] for h in padding_hits[gi]))
        print(f"group {gi}: pad region = {pad_bytes} bytes/file, "
              f"{n_files} files show scribble there (expected if unused)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
