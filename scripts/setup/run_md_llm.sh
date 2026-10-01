#!/usr/bin/env bash
# Gemma text generation for FLAG-MD subgraphs, sharded across GPUs, resumable.
#
# FLAG-MD changes which nodes are in each subgraph, and Gemma's text is generated
# per subgraph, so it needs its own LLM cache. Subgraphs whose node list is
# IDENTICAL to the cosine sampler's are reused (--reuse-from-cosine), so only the
# differing ones are generated. Work is split into small shards so a crash costs
# at most one shard; finished shards are skipped on re-run.
#
#   bash scripts/setup/run_md_llm.sh                     # all datasets/kinds, K=2 matched
#   WORKERS_PER_GPU=3 SHARDS=12 bash scripts/setup/run_md_llm.sh
#   DATASETS="reddit" KINDS="discriminative" bash scripts/setup/run_md_llm.sh
#   MD_ARGS="--diffusion-steps 3" bash scripts/setup/run_md_llm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONWARNINGS=ignore
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-.venv-gpu/bin/python}"

DATASETS="${DATASETS:-reddit instagram}"
KINDS="${KINDS:-discriminative residual}"
SHARDS="${SHARDS:-12}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"   # each worker peaks at ~30 GB (fp16 logits over long prompts); 3/GPU OOMs
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')}"
MD_ARGS="${MD_ARGS:---diffusion-steps 2 --md-selection matched_cosine}"
# Same decode budget as the production cosine cache (decision D-004) -- REQUIRED
# for the reuse to be valid and for the comparison to be fair.
GEN_ARGS="${GEN_ARGS:---max-new-tokens 64 --truncate-chars 300}"
LOGDIR="logs/llm"; mkdir -p "$LOGDIR"

JOBS=$(mktemp)
for ds in $DATASETS; do for kind in $KINDS; do
  for ((i=0; i<SHARDS; i++)); do echo "$ds $kind $i"; done
done; done > "$JOBS"

run_one() {  # gpu dataset kind shard
  local gpu=$1 ds=$2 kind=$3 shard=$4
  $PY -m scripts.llm.generate_text --dataset "$ds" --kind "$kind" \
      --strategy markov_diffusion $MD_ARGS $GEN_ARGS --reuse-from-cosine \
      --num-shards "$SHARDS" --shard "$shard" --device "cuda:$gpu" \
      > "$LOGDIR/md_${ds}_${kind}_shard${shard}.log" 2>&1 \
    && echo "done  $ds $kind shard $shard (gpu $gpu)" \
    || echo "FAIL  $ds $kind shard $shard (gpu $gpu) -> $LOGDIR/md_${ds}_${kind}_shard${shard}.log"
}
export -f run_one
export PY MD_ARGS GEN_ARGS SHARDS LOGDIR

# Two passes: finished shards are skipped, so pass 2 only retries failures (e.g. OOM).
for pass in 1 2; do
  echo "== pass $pass =="
  for ((g=0; g<NUM_GPUS; g++)); do
    awk -v n="$NUM_GPUS" -v g="$g" '(NR-1)%n==g' "$JOBS" \
      | xargs -P "$WORKERS_PER_GPU" -L1 bash -c 'run_one '"$g"' "$@"' _ &
  done
  wait
done

echo "all shards attempted; merging"
for ds in $DATASETS; do for kind in $KINDS; do
  $PY -m scripts.llm.generate_text --dataset "$ds" --kind "$kind" \
      --strategy markov_diffusion $MD_ARGS $GEN_ARGS --reuse-from-cosine \
      --num-shards "$SHARDS" --merge --device cuda:0 2>&1 | tail -6
done; done
