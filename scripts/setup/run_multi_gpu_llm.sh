#!/usr/bin/env bash
# Parallel multi-GPU LLM text generation (decision D-003 stage).
#
# scripts/llm/generate_text.py loops over --dataset {reddit,instagram} x
# --kind {discriminative,residual} sequentially on one GPU. Those four
# (dataset, kind) jobs are fully independent -- each reads its own
# subgraphs/prompts and writes its own cache/llm/<cache_key>.json +
# .manifest.json, no shared state -- so with more than one GPU they can just
# run at the same time instead of one after another.
#
# This does NOT shard the LLM model itself across GPUs (gemma-2-9b-it is
# ~18.5 GB fp16 and fits on a single 24+ GB card; sharding it would be
# DDP/NCCL complexity for zero benefit here). Each GPU runs its own full copy
# of the model, one job at a time, so N GPUs never load more than N copies
# at once regardless of how many jobs there are.
#
# Usage (from repo root, on the GPU instance, after install_gpu.sh and
# fetch_methods.sh have already been run):
#   bash scripts/setup/run_multi_gpu_llm.sh
#   DATASET=reddit bash scripts/setup/run_multi_gpu_llm.sh
#   DATASET=instagram bash scripts/setup/run_multi_gpu_llm.sh
#   NUM_GPUS=2 bash scripts/setup/run_multi_gpu_llm.sh        # override autodetect
#   GENERATE_ARGS="--limit 20" bash scripts/setup/run_multi_gpu_llm.sh   # smoke test first
#
# With DATASET=reddit and 2 GPUs, Reddit discriminative and residual generation
# run at the same time, one job per GPU. With 4 GPUs and DATASET=all, all four
# dataset/kind jobs run at once. With fewer GPUs, jobs are grouped onto each
# GPU and run sequentially within that GPU -- never more than one job per GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# The benchmark and sampling caches are trusted PyG Data objects. This keeps
# PyTorch 2.6+ from applying weights_only=True to those legacy artifacts.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

PYTHON="${PYTHON:-python}"
if [ -x ".venv-gpu/bin/python" ]; then
  PYTHON=".venv-gpu/bin/python"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found -- this does not look like a GPU machine."
  echo "  See docs/vastai_gpu_workflow.md."
  exit 1
fi

DETECTED_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
NUM_GPUS="${NUM_GPUS:-$DETECTED_GPUS}"
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: no GPUs detected."
  exit 1
fi

DATASET="${DATASET:-all}"
case "$DATASET" in
  reddit|instagram|all) ;;
  *)
    echo "ERROR: DATASET must be reddit, instagram, or all; got: $DATASET"
    exit 2
    ;;
esac

echo "=============================================================="
echo "Multi-GPU LLM generation"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  using $NUM_GPUS of the above GPU(s)"
echo

# Independent (dataset, kind) jobs. DATASET=reddit or DATASET=instagram is
# useful when exactly two GPUs should process both kinds of one dataset at once.
if [ "$DATASET" = "all" ]; then
  JOBS=(
    "reddit discriminative"
    "reddit residual"
    "instagram discriminative"
    "instagram residual"
  )
else
  JOBS=(
    "$DATASET discriminative"
    "$DATASET residual"
  )
fi

mkdir -p logs/llm
pids=()

for g in $(seq 0 $((NUM_GPUS - 1))); do
  assigned=()
  for i in "${!JOBS[@]}"; do
    if [ $(( i % NUM_GPUS )) -eq "$g" ]; then
      assigned+=("${JOBS[$i]}")
    fi
  done
  [ "${#assigned[@]}" -eq 0 ] && continue

  echo "cuda:$g will run, in order: ${assigned[*]}"

  (
    for job in "${assigned[@]}"; do
      read -r dataset kind <<< "$job"
      log="logs/llm/${dataset}_${kind}.log"
      echo "[cuda:$g] starting $dataset/$kind (log: $log)"
      # shellcheck disable=SC2086
      if ! "$PYTHON" -m scripts.llm.generate_text \
            --dataset "$dataset" --kind "$kind" --device "cuda:${g}" \
            ${GENERATE_ARGS:-} > "$log" 2>&1; then
        echo "[cuda:$g] FAILED $dataset/$kind -- see $log" >&2
        exit 1
      fi
      echo "[cuda:$g] finished $dataset/$kind"
    done
  ) &
  pids+=("$!")
done

echo
echo "waiting for ${#pids[@]} GPU worker(s)..."
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

if [ "$failed" -ne 0 ]; then
  echo
  echo "One or more jobs failed. Check logs/llm/*.log before trusting the cache."
  exit 1
fi

echo
echo "All jobs completed. Verifying generated LLM cache checksums..."
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

manifests = sorted(Path("cache/llm").glob("*.manifest.json"))
if not manifests:
    raise SystemExit("No LLM manifests were generated")

for manifest_path in manifests:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = Path(manifest["output"])
    actual = hashlib.sha256(output_path.read_bytes()).hexdigest()
    expected = manifest["output_sha256"]
    coverage = manifest["stats"]["node_coverage"] * 100
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {output_path}")
    print(f"OK {output_path} | coverage={coverage:.1f}%")
PY

echo
echo "Done. The generated cache is in cache/llm/."
