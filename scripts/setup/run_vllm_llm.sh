#!/usr/bin/env bash
# Batched LLM generation with vLLM across every visible GPU (data parallel).
#
#   bash scripts/setup/run_vllm_llm.sh                # cosine, then FLAG-MD, then encode
#   STAGES="cosine" bash scripts/setup/run_vllm_llm.sh
#
# One process per GPU (CUDA_VISIBLE_DEVICES=g). Each loads Gemma once and runs its
# 1/N stride of every (dataset, kind) job, writing *.part<g>of<N>.json; a merge
# then writes the final cache exactly as the HF path does. Resumable: a finished
# shard file is skipped. FLAG-MD runs after cosine because it reuses the cosine
# text for every subgraph whose node list is identical (--reuse-from-cosine).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false

PY="${PYTHON:-.venv-vllm/bin/python}"
DATASETS="${DATASETS:-reddit instagram}"
KINDS="${KINDS:-discriminative residual}"
STAGES="${STAGES:-cosine md encode}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')}"
GEN_ARGS="${GEN_ARGS:---engine vllm --dtype bfloat16 --max-new-tokens 550 --truncate-chars 1200}"
MD_ARGS="${MD_ARGS:---strategy markov_diffusion --diffusion-steps 2 --md-selection matched_cosine --reuse-from-cosine}"
LOGDIR="logs/llm"; mkdir -p "$LOGDIR"

generate_stage() {  # name, extra args
  local name=$1; shift
  local ds_list kinds_arg
  ds_list=$(echo $DATASETS | tr ' ' ',')
  kinds_arg=both; [ "$(echo $KINDS | wc -w)" -eq 1 ] && kinds_arg=$KINDS
  echo "== $name: $NUM_GPUS GPU workers, datasets=$ds_list kind=$kinds_arg =="
  local pids=()
  for ((g=0; g<NUM_GPUS; g++)); do
    # One process per GPU: Gemma is loaded once and serves every (dataset, kind).
    CUDA_VISIBLE_DEVICES=$g $PY -P -m scripts.llm.generate_text --dataset "$ds_list" \
      --kind "$kinds_arg" $GEN_ARGS "$@" --num-shards "$NUM_GPUS" --shard "$g" \
      --device cuda:0 >> "$LOGDIR/vllm_${name}_gpu${g}.log" 2>&1 &
    pids+=("$!")
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 0 ] || { echo "$name: a worker failed -- see $LOGDIR/vllm_${name}_gpu*.log"; exit 1; }
  # Merge: no model load, just stitches the shard files (+ reused text) together.
  CUDA_VISIBLE_DEVICES=0 $PY -P -m scripts.llm.generate_text --dataset "$ds_list" --kind "$kinds_arg" \
      $GEN_ARGS "$@" --num-shards "$NUM_GPUS" --merge --device cuda:0 2>&1 \
      | grep -E "^[A-Z]{4,}|generated|format fails|coverage|wrote" || true
}

for stage in $STAGES; do
  case "$stage" in
    cosine) generate_stage cosine --strategy semantic ;;
    md)     generate_stage md $MD_ARGS ;;
    encode)
      echo "== encode: Sentence-BERT of every new cache =="
      i=0
      for ds in $DATASETS; do
        for strat in "--strategy semantic" "$MD_ARGS"; do
          strat_enc="${strat/--reuse-from-cosine/}"
          CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) $PY -P -m scripts.preprocess.encode_llm_text \
            --dataset "$ds" --kind both \
            $strat_enc --device cuda:0 --batch-size 256 > "$LOGDIR/encode_${ds}_${i}.log" 2>&1 &
          i=$((i+1))
        done
      done
      wait
      tail -n 4 "$LOGDIR"/encode_*.log
      ;;
    *) echo "unknown stage $stage"; exit 2 ;;
  esac
done
