#!/bin/bash
# Rebuild the installed vLLM to match this working tree's patches/ + kvarn/.
#
# Lets you `git checkout <any commit>` in THIS repo and have the venv's vLLM
# python files follow, so history can be stepped through / bisected. Needs the
# installed package to be a git repo with a `pristine-0.27.1` branch -- see
# docs/vllm-git.md.
#
#   bash sync-venv.sh              rebuild venv to match the current checkout
#   bash sync-venv.sh --check      report drift, change nothing (exit 1 if stale)
#   bash sync-venv.sh --install-hook   auto-run on every git checkout
#
# Only .py files are touched. The 577 MB of compiled .so from the wheel are
# untracked+ignored inside the venv repo, so no git operation here can remove
# them (nothing uses `git clean -x`, deliberately).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=${PY:-$HERE/venv/bin/python}
SP=$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null | tail -n1)
[ -n "$SP" ] && [ -d "$SP" ] || { echo "sync-venv: cannot import vllm with $PY"; exit 1; }
[ -d "$SP/.git" ] || { echo "sync-venv: $SP is not git-tracked (see docs/vllm-git.md)"; exit 1; }
git -C "$SP" rev-parse --verify -q pristine-0.27.1 >/dev/null || {
  echo "sync-venv: no 'pristine-0.27.1' branch in $SP (see docs/vllm-git.md)"; exit 1; }

WANT="$(git rev-parse HEAD)"
STAMP="$SP/.git/synced-from"
MODE="${1:-}"

if [ "$MODE" = "--install-hook" ]; then
  mkdir -p "$HERE/.git/hooks"
  cat > "$HERE/.git/hooks/post-checkout" <<'HOOK'
#!/bin/sh
# installed by sync-venv.sh --install-hook
# $3 == 1 for a branch checkout, 0 for a file checkout; only act on the former.
[ "$3" = "1" ] || exit 0
exec bash "$(git rev-parse --show-toplevel)/sync-venv.sh" || true
HOOK
  chmod +x "$HERE/.git/hooks/post-checkout"
  echo "sync-venv: installed .git/hooks/post-checkout"
  echo "  NOTE hooks are local to this clone and are not pushed; re-run after re-cloning."
  exit 0
fi

if [ "$MODE" = "--check" ]; then
  have=$(cat "$STAMP" 2>/dev/null || echo none)
  if [ "$have" = "$WANT" ]; then echo "sync-venv: venv matches $(git log -1 --format=%h)"; exit 0; fi
  echo "sync-venv: venv built from ${have:0:12}, working tree is ${WANT:0:12} -- run: bash sync-venv.sh"
  exit 1
fi

# Refuse to silently discard hand edits inside the venv. They are only
# recoverable if they were committed there or captured as a patch.
if [ -n "$(git -C "$SP" status --porcelain)" ]; then
  echo "sync-venv: REFUSING -- uncommitted edits in $SP would be lost:"
  git -C "$SP" status --porcelain | head -10 | sed 's/^/    /'
  echo "  capture them first (regenerate the patch + commit there), or discard with:"
  echo "    git -C $SP checkout -- ."
  exit 1
fi

echo "sync-venv: rebuilding vLLM from patches/ at $(git log -1 --format='%h %s' | cut -c1-60)"
git -C "$SP" checkout -q --detach pristine-0.27.1
git -C "$SP" reset -q --hard pristine-0.27.1     # never `clean -x`: would delete the .so

# ORDER MATTERS, and the right order is this repo's own history -- the sequence
# in which the patches were originally written and applied. The README's plain
# `for p in patches/*.patch` glob is alphabetical, which fails on ~6 of them
# (e.g. mamba-align-stale-checkpoint-offload-slack sorts BEFORE the
# mamba-align-stale-state-queue it builds on). Verified 2026-08-23: replaying in
# first-appearance-in-history order applies 26/27 in a single pass with no
# retries, and reproduces the deployed venv exactly.
#
# --diff-filter=A --name-only over `git log --reverse` gives each patch's
# first-appearance commit; awk '!seen[$0]++' keeps the first occurrence only.
# Patches deleted since (discard-unoffloaded-kv-warning, the verbose offload
# logger, ...) still appear in history, so skip any that no longer exist.
order=$(git log --reverse --format='%H' --diff-filter=A --name-only -- patches/ 2>/dev/null \
        | grep '^patches/.*\.patch$' | awk '!seen[$0]++')
# Anything present but never seen in history (uncommitted new patch) goes last.
for p in patches/*.patch; do
  printf '%s\n' "$order" | grep -qxF "$p" || order="$order
$p"
done

applied=0; pending=""
for p in $order; do
  [ -f "$p" ] || continue
  if patch -p1 -s --dry-run -d "$SP" < "$p" >/dev/null 2>&1; then
    patch -p1 -s -d "$SP" < "$p" >/dev/null 2>&1
    applied=$((applied+1))
  else
    pending="$pending $p"
  fi
done
# Second chance for anything that failed: a patch whose own file was edited
# after it was first added can need a later position than first-appearance.
if [ -n "$pending" ]; then
  progress=1
  while [ -n "$pending" ] && [ -n "$progress" ]; do
    progress=""; still=""
    for p in $pending; do
      if patch -p1 -s --dry-run -d "$SP" < "$p" >/dev/null 2>&1; then
        patch -p1 -s -d "$SP" < "$p" >/dev/null 2>&1
        applied=$((applied+1)); progress=1
      else still="$still $p"; fi
    done
    pending="$still"
  done
fi
find "$SP" \( -name '*.rej' -o -name '*.orig' \) -delete 2>/dev/null || true

if [ -f kvarn/kvarn-0.27.1.patch ] && [ -d kvarn/files/vllm ]; then
  cp -r kvarn/files/vllm/. "$SP/"
  patch -p1 -N -r /dev/null -s -d "$SP" < kvarn/kvarn-0.27.1.patch >/dev/null 2>&1 || true
fi
find "$SP" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

git -C "$SP" add -A >/dev/null
git -C "$SP" commit -q -m "synced from $(basename "$HERE") $(git log -1 --format='%h %s' | cut -c1-60)" >/dev/null 2>&1 || true
echo "$WANT" > "$STAMP"

echo "sync-venv: applied $applied patch(es)"
if [ -n "$pending" ]; then
  echo "sync-venv: could NOT apply (this commit's patches/ do not build cleanly):"
  for p in $pending; do echo "    $(basename "$p")"; done
fi
"$PY" -c 'import vllm; print("sync-venv: import ok,", vllm.__version__)'
