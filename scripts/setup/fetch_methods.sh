#!/usr/bin/env bash
# Reconstruct methods/ from upstream at pinned commits.
#
# methods/ is git-ignored: four of these repositories ship NO LICENSE file, so
# this project must not re-host their source. Provenance is preserved by pinning
# exact SHAs here instead of vendoring code.
#
# Usage:
#   bash scripts/setup/fetch_methods.sh            # fetch everything
#   bash scripts/setup/fetch_methods.sh flag       # fetch one
#   bash scripts/setup/fetch_methods.sh --verify   # check existing clones only
#
# See research/repository_provenance.md for classification and licence notes.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHODS="$ROOT/methods"

# name|url|commit_sha|licence
# A commit of "HEAD" means we have not pinned it yet and must record what we get.
REPOS=(
  "flag|https://github.com/BUPT-GAMMA/FLAG.git|cb83944ed8a8a9b070a3f5a167d363973369fc80|NONE"
  "care_gnn|https://github.com/YingtongDou/CARE-GNN.git|a64ff7523e187a24251f7ca88435d2c9d8f7dcd9|Apache-2.0"
  "bwgnn|https://github.com/squareRoot3/Rethinking-Anomaly-Detection.git|de0631f039bbd19c1890b483cc01f1007f596af7|NONE"
  "dga_gnn|https://github.com/AtwoodDuan/DGA-GNN.git|0907392f6060e18230339ca30eaca0c917414820|NONE"
  "pmp|https://github.com/Xtra-Computing/PMP.git|3f7629f6c180891a0bc1bba3c66d94d288a1ddae|NONE"
  "geniepath|https://github.com/shuowang-ai/GeniePath-pytorch.git|143f07cc49ef9eb9fe176cc355ba2ccba609e57a|MIT"
  "glbench|https://github.com/NineAbyss/GLBench.git|ec8be30287872fa65c6d0619cffcbb5ffaed2e72|MIT"
)

# GCN and GAT are intentionally absent: we use PyG's GCNConv/GATConv rather than
# the authors' TensorFlow-1 repositories. tkipf/pygcn's own README disclaims
# reproduction, and GAT's author explicitly recommends PyG/DGL instead. Rationale
# in research/repository_provenance.md sections 7-8.

VERIFY_ONLY=0
WANTED=""
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY_ONLY=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) WANTED="$arg" ;;
  esac
done

mkdir -p "$METHODS"
fail=0

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url sha licence <<< "$entry"
  [ -n "$WANTED" ] && [ "$WANTED" != "$name" ] && continue

  dest="$METHODS/$name"
  printf '=== %-12s %s\n' "$name" "$url"

  if [ -d "$dest/.git" ]; then
    actual="$(git -C "$dest" rev-parse HEAD 2>/dev/null)"
    if [ "$sha" = "HEAD" ]; then
      echo "    present, unpinned, at $actual"
      echo "    ACTION: pin this SHA in fetch_methods.sh and repository_provenance.md"
    elif [ "$actual" = "$sha" ]; then
      echo "    OK at pinned $sha"
    else
      echo "    MISMATCH"
      echo "      expected $sha"
      echo "      actual   $actual"
      echo "    Upstream findings in research/ may be stale. Re-run the audit tests."
      fail=1
    fi
    # Ignore __pycache__: importing upstream modules for the audit tests creates
    # it, and that is not a source modification.
    dirty="$(git -C "$dest" status --porcelain 2>/dev/null | grep -v '__pycache__' || true)"
    if [ -n "$dirty" ]; then
      echo "    WARNING: local modifications present. Upstream must stay pristine;"
      echo "             adaptations belong in src/adapters/, not here."
      git -C "$dest" status --short | sed 's/^/               /'
    fi
    echo "    licence: $licence"
    continue
  fi

  if [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "    MISSING (verify-only mode, not cloning)"
    fail=1
    continue
  fi

  if ! git clone --quiet "$url" "$dest"; then
    echo "    CLONE FAILED"
    fail=1
    continue
  fi
  if [ "$sha" != "HEAD" ]; then
    if ! git -C "$dest" checkout --quiet "$sha" 2>/dev/null; then
      echo "    CHECKOUT of $sha FAILED (commit gone from upstream?)"
      fail=1
      continue
    fi
  fi
  echo "    cloned at $(git -C "$dest" rev-parse HEAD)"
  echo "    licence: $licence"

  if [ "$licence" = "NONE" ]; then
    echo "    NOTE: no licence upstream -> local research use only."
    echo "          Do NOT redistribute. See repository_provenance.md section 11."
  fi
done

echo
if [ "$fail" -ne 0 ]; then
  echo "FAILED - see messages above."
  exit 1
fi
echo "All requested repositories present at their pinned commits."
echo
echo "Reminder: four of these ship no LICENSE (flag, bwgnn, dga_gnn, pmp)."
echo "methods/ is git-ignored so this project never re-hosts them."
