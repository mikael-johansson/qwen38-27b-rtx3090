# KV-offload restore corruption (OPEN) + the cache-identity bug found chasing it

**Status 2026-08-24:**
- **OPEN:** a conversation restored from the KV offload tier decodes token
  soup. Reproduces deterministically. **Not fixed.**
- **FIXED:** a separate, genuine defect found during the investigation — the
  offload cache directory identity ignored the Mamba state layout
  (`patches/kv-offload-cache-identity-mamba-state.patch`). Real bug, worth
  keeping, but it does **not** fix the symptom above.

**Regression test:** `test_kv_cache_identity_corruption.py`

## Symptom

A conversation returns fluent-looking multilingual token soup, or a run of
`!!!!!!`, from the very first decoded token:

```
<think>IBE大法没说rognyderencer如其telephone霞\E_colavourembreAMPL斗争истраukup RELA…
<think>_unc!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Everything else looks healthy. From the 2026-08-24 06:50:33 case: 88,177
prompt tokens, 87,360 cached (99.1%), 1.29s TTFT, 20 tok/s, no warning
anywhere. A request two seconds later, same prompt shape, was coherent.

The tell is that it is **per conversation, not per server**: within one
server, the conversation that stays GPU-resident is fine indefinitely while
the one that gets evicted and restored is garbage on every turn.

## Reproduction (deterministic)

`test_kv_cache_identity_corruption.py` — two ~88K-token conversations,
alternating turns. Only about two fit in the 6.11 GiB GPU pool, so one stays
resident while the other is evicted and restored:

```
r0 c0  prompt= 58669   0.9s  'ready'
r0 c1  prompt= 58672   5.2s  '联ditigar-addon령ester上空克拉求精vik…'   <<< CORRUPTED
r1 c0  prompt= 58693   1.1s  'ok'
r1 c1  prompt= 58820   6.0s  '克森樣刘晓_par脊至少有 quỹdocaringlick…'   <<< CORRUPTED
```

4/4 on the restored conversation. The resident one (~1.1s/turn) is never
affected; the restored one (5–6s/turn) always is.

Verified with a **clean, single-config cache**, the identity fix applied, and
`fs -> RAM` promotions **zero** — i.e. this instance was a pure **RAM-tier**
restore. The disk tier is not required to trigger it.

## What is established

| configuration | result |
|---|---|
| offload connector **disabled** | **0/8 corrupted** (evicted prefix is recomputed, ~65s) |
| offload enabled, conversation stays GPU-resident | always clean |
| offload enabled, conversation restored | corrupted, 4/4 |

Disabling the offload connector is the only configuration proven clean. That
is the workaround, at the cost of recomputing an evicted prefix (~65s per 88K
conversation).

`grep "GPU had 0 tok" qwen.log` finds full restores — the case where the whole
prefix comes back from the tier. Those are the ones that corrupt. The
`KV LOAD` / `KV STORE` lines from `patches/kv-transfer-request-logging.patch`
make this legible in one line:

```
KV LOAD chatcmpl-b094b3280e079f1b | fs   -> RAM  | 284 blk
KV LOAD chatcmpl-b094b3280e079f1b | RAM  -> VRAM |  71 blk / 59072 tok (GPU had 0 tok)
```

3 of this model's 4 KV groups are `MambaManager` (48 `linear_attn` layers), so
three quarters of a restored prefix is recurrent state — the part with no
per-token redundancy to absorb an error.

## Hypotheses ruled out

Each of these looked convincing and cost real time. Recorded so they are not
re-tried:

1. **`--long-prefill-token-threshold 832`.** The strongest early signal:
   garbage began 6 minutes into the first 832 server (08-23 12:03) and the one
   512 window that day was clean over 20 requests. But 832 drove **433 offload
   lookups** in its window against **57** at 512 — it raises exposure to
   restores, it does not cause the corruption. Both thresholds reproduce.
2. **The vision tower.** Timing rules it out: first garbage 08-23 12:10:06,
   first vision commit 12:22:56 — 12 minutes later.
3. **The align-mode Mamba free path** freeing a live checkpoint block.
   `MAMBA_FREE` fired **0 times** during corrupting runs; the path is never
   entered. Both candidate off-by-ones (`cdiv(processed, 832) - 1` and
   `(processed - 1) // 832`) are correctly conservative at exact multiples.
4. **O_DIRECT misalignment** in the fs tier. Block files are exactly
   4096-aligned (28,966,912 bytes); buffered and direct I/O behave the same.
   And the RAM-only repro above involves no disk I/O at all.
5. **`fs-tier-disk-cap.patch` deleting live blocks.** Zero cap evictions
   during the corrupting runs (disk at 40%).
6. **The NVMe tier specifically.** Reproduced with `fs -> RAM: 0`.
7. **A poisoned mixed-config cache** (see below). Real, and fixed — but the
   corruption still reproduces on a clean single-config cache.

## Caution about "clean" runs

Several configurations looked clean early on and later reproduced the bug —
including RAM-tier-only, which showed 0/8 with 5 full restores and afterwards
failed 4/4. Those early passes were **low exposure, not proof**: when both
conversations happened to stay resident, turns take ~1.1s and nothing is
restored. Always check the restore actually happened (`GPU had 0 tok`, and a
turn taking 5s+ rather than ~1s) before reading a pass as meaningful.

## The separate bug that was found and fixed

The offload cache's directory is `<root_dir>/<model>_<sha256-prefix>`, hashed
over `FileMapper.fields`. That set contained:

```
blocks_per_file, dcp_size, dtype, inference_engine, model_name,
kv_cache_groups[{layer_names, tokens_per_block}], parallel_agnostic,
pcp_size, pp_size, tokens_per_hash, tp_size
```

`dtype` is the **attention** KV dtype (`fp8` here). Nothing described the
recurrent state. So two servers differing only in `--mamba-ssm-cache-dtype`
(fp16 / bf16 / fp32) or `--mamba-cache-mode` hashed to the **same directory**
and would load each other's blocks, reinterpreting the bytes. Not a cache miss
— silent corruption, with no checksum or version tag to catch it.

Directory `…AutoRound_1173ed6e0f24` here had accumulated blocks across several
days of fp16/bf16/fp32 experiments and several vLLM builds. `PYTHONHASHSEED=0`
(see `micke-start.sh`) makes block hashes reproducible across restarts — which
is what makes the disk cache useful, and also what lets stale incompatible
blocks be found and loaded after a config change.

**Fix:** `mamba_ssm_cache_dtype` and `mamba_cache_mode` are threaded from
`vllm_config.cache_config` through `OffloadingCacheConfig` into
`FileMapper.fields`. Written only when non-empty and only for models that
actually have a non-`AttentionSpec` group, following the existing
`replicated_layout` convention — so models with no recurrent groups keep their
exact current directory hash and cache. The vLLM build is deliberately not
hashed: stricter, but it would force a cold cache on every patch applied here.

Verified: fp16 and fp32 now hash to different directories, `mamba_cache_mode`
likewise, the same config is stable across runs, a model with no mamba groups
is byte-identical to before, and the patched build correctly ignored the old
directory (65s recompute) and minted a new one recording
`{"mamba_cache_mode": "align", "mamba_ssm_cache_dtype": "float16"}`.

This closes a real latent bug — mixed SSM dtypes sharing one directory is
indefensible regardless — but it is **not** the cause of the open symptom.

## Where to look next

The restore path writes recurrent state back into GPU blocks. Worth
investigating, roughly in order:

1. Whether the restored SSM state corresponds to the **end** of the last
   restored block or some other boundary — an off-by-one-block here would
   produce exactly this (coherent prefix, wrong state, immediate divergence).
2. Whether `mamba_cache_mode=align`'s block-boundary checkpoint is what
   actually gets stored, or whether a mid-block state is captured.
3. Whether the resident-vs-restored asymmetry is about *which* block supplies
   the resume state rather than about the transfer itself — the bytes survive
   a RAM round trip fine, so the likely defect is in selecting/interpreting
   the resume checkpoint, not in moving it.
4. A direct check: checksum a group's SSM state at store time and at restore
   time and compare. That distinguishes "wrong bytes" from "right bytes, wrong
   block".
