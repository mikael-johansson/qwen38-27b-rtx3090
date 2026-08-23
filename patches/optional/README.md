# Optional / not-applied patches

`verify.sh` requires every `patches/*.patch` to be applied to the venv. Patches
in here are deliberately **not** applied — they are diagnostic tools kept for
when an investigation needs them again. Moving one back up a directory is not
enough; apply it explicitly and restart:

```sh
patch -p1 -d venv/lib/python3.12/site-packages/vllm < patches/optional/<name>.patch
```

## offload-verbose-eviction-debug-logging.patch

Extremely verbose `logger.debug()` tracing across the KV-offload connector's
store/lookup/eviction paths, plus block-level `EVICT` / `REALLOC` / `FLUSH` /
`MAMBA_FREE` / `STORE_JOB_BLOCKS` lines. All tagged `request-logging
offload-verbose:` for grepping.

Written for the 2026-08-22/23 HIT_DIVERGED investigation
(see `docs/mamba-align-prefill-leak.md`). Reverted 2026-08-23 once that closed:
it was ~75% of all log lines (324k of ~430k in a 55-minute window, ~100 MB/hour).
With it reverted and `logging-to-file.json` (INFO) in use, the same workload
logs ~275 KB/hour.

Needs `VLLM_LOGGING_CONFIG_PATH=single-user/logging-to-file-debug.json` to be
useful, since every line it adds is `DEBUG`.

The one production-relevant signal it used to carry — `HIT_DIVERGED RECONCILE`,
i.e. a deep prefix hit was thrown away — was promoted to `logger.warning` in
`patches/hit-diverged-boundary-rescue.patch`, so it stays visible at INFO
without this patch.
