#!/usr/bin/env bash
# Main 5x5 push (decision D-005): flag + flag_finetuned, each with FLAG's cosine
# sampler and FLAG-MD (md_K2_matched), on every dataset, on CPU.
#
#   bash scripts/reproduce/run_main_flag.sh
#
# Results: results/main/<variant>/<sampler>/. Resumable (see run_flag_md_matrix.sh).
# Training replays the cached LLM embeddings only; no GPU is needed, and CUDA is
# hidden so the CPU workers do not each open a context on GPU 0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
DATASETS="${DATASETS:-reddit instagram amazon_text yelpchi_text}"
PAR="${PAR:-28}"          # per variant; 2 variants x 28 x 2 threads ~= the 122-core quota
mkdir -p logs/main

pids=()
for variant in flag flag_finetuned; do
  env VARIANT=$variant SAMPLERS="cosine md_K2_matched" DATASETS="$DATASETS" \
      SEEDS=5 INITS=5 DEVICE=cpu PAR="$PAR" OUT="results/main/$variant" \
      bash scripts/reproduce/run_flag_md_matrix.sh > "logs/main/${variant}_matrix.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
echo "flag + flag_finetuned matrices finished"
grep -h "FAIL" logs/main/flag_matrix.log logs/main/flag_finetuned_matrix.log || echo "no failed jobs"
