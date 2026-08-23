# Git-tracking the installed vLLM

The installed package at `venv/lib/python3.12/site-packages/vllm/` is its own
git repository, so you can step between states with `git checkout` / `git bisect`
instead of reverse-applying patch files by hand.

Set up 2026-08-23. `verify.sh` checks it stays in sync.

[← back to the main README](../README.md)

## Why not a fork, a submodule, or a source install

vLLM here is a **prebuilt wheel** (`vllm-0.27.1.dist-info`) with **577 MB of
compiled `.so`** across 17 CUDA extensions.

- **Source install / fork** (`pip install -e .` against a clone) is what "fork
  it" normally implies. It means recompiling those kernels: hours of build, an
  exactly-matching toolchain, and *different binaries* than the wheel this
  repo's numbers were measured against. Given the Mamba-kernel numerical
  problems chased on 2026-08-22/23, silently swapping compiled kernels under an
  investigation is a good way to waste a day.
- **Git submodule** points at a git repo; `site-packages/vllm` isn't one, so
  this reduces to the source-install problem.
- **A fork is not needed at all** for stepping through changes. Forking is a
  GitHub-side concept for pushing branches and opening PRs. Local branches give
  100% of the bisect capability. Fork only if you intend to upstream something.

So: keep the wheel exactly as installed, and put git *around* it.

## What is tracked

Only `.py`. Binaries are excluded in `.git/info/exclude` (not a tracked
`.gitignore`, so the rules can never leak into a generated patch):

```
total       750M
  .so       577M   excluded — from the wheel, never patched
  pycache    21M   excluded
  .py        39M   tracked  — every patch in patches/ touches only .py
```

Resulting repo: ~50 MB.

## The rule: patches/ and this repo must agree

`patches/*.patch` stays the **source of truth** — `verify.sh`, the README and
the Docker build all depend on it, and it is the portable artifact. The git repo
is a *working tool* on top of it.

**Every edit to the installed vLLM must end up in both places.** A change that
exists only in the working tree is lost by the next `pip install`, reverse-apply
or `git checkout`.

```sh
SP=venv/lib/python3.12/site-packages/vllm

# 1. edit the file(s) under $SP as usual, then:
git -C $SP diff -- v1/core/sched/scheduler.py     # regenerate the patch body
#    ...paste under the docstring in patches/<name>.patch, keeping a/ b/ prefixes

# 2. confirm the patch matches the tree (this is what verify.sh runs):
patch -p1 -R --dry-run -s -d $SP < patches/<name>.patch && echo "patch matches tree"

# 3. commit, referencing the patch file by name:
git -C $SP add -A
git -C $SP commit -m "<name>.patch: one-line summary"
```

`bash verify.sh` fails if the tree is dirty, so drift gets caught on the next
run rather than weeks later:

```
FAIL  vllm git tree DIRTY — uncommitted edits in ... not captured in patches/ or a commit:
         M envs.py
```

## What this buys

- `git checkout <commit>` / `git bisect` across states — no hand reverse-applying.
- **Safe experiments.** Reverse-applying a patch to test a hypothesis used to
  risk leaving `.rej` files and a half-patched tree (there are still stale
  `qwen3_5.py.rej` / `qwen3_5_mtp.py.rej` from an older attempt). Now
  `git checkout -- .` restores the exact deployed state.
- `git diff` regenerates any patch body.
- `git stash` to park an experiment while serving traffic from a known state.

## Limits, honestly

- **This is not upstream history.** The baseline commit is *upstream 0.27.1 plus
  every applied patch*, squashed — not pristine upstream. You can bisect changes
  made from 2026-08-23 onward, not the patches that predate it.
  To get per-patch history you would need a pristine 0.27.1 tree (re-download
  the wheel into a scratch dir), commit that as the root, then apply
  `patches/*.patch` one at a time committing each. Worth doing if a regression
  ever needs to be bisected across the existing patch set; not done here because
  reverse-applying ~21 overlapping patches against the *live* server directory
  is exactly the risky operation this setup exists to avoid.
- **A `pip install --force-reinstall vllm` deletes `.git`.** Re-run the setup
  (below) afterwards.
- The nested repo lives inside a directory the outer repo ignores
  (`.gitignore` line 8, `venv/`), so the two never interact.

## Recreating it

```sh
cd venv/lib/python3.12/site-packages/vllm
git init -q
printf '*.so\n*.pyd\n*.pyc\n__pycache__/\n*.rej\n*.orig\n' > .git/info/exclude
git add -A
git commit -m "vllm 0.27.1 wheel, as currently deployed (all patches/*.patch applied)"
git tag baseline-$(date +%Y-%m-%d)
```
