#!/usr/bin/env bash
# Sampler comparison matrix: FLAG (cosine) vs FLAG-MD (Markov diffusion).
#
#   VARIANT=text  -> LLM-free: same GNN + raw-text features, only the sampler differs.
#   VARIANT=flag  -> full FLAG pipeline (needs the sampler's Gemma cache).
#
# Every run is (dataset, backbone, sampler, seed 0..SEEDS-1, init 0..INITS-1). The
# seed/init streams do not depend on the sampler, so cosine and MD runs are PAIRED.
# Resumable: a (dataset, model, sampler) job with all its result files is skipped.
#
#   VARIANT=text bash scripts/reproduce/run_flag_md_matrix.sh
#   VARIANT=flag SAMPLERS="cosine md_K2_matched" DEVICE=cuda:0 PAR=4 bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}"
PY="${PYTHON:-.venv-gpu/bin/python}"

VARIANT="${VARIANT:-text}"
DATASETS="${DATASETS:-reddit instagram}"
MODELS="${MODELS:-geniepath care_gnn gat bwgnn dga_gnn pmp gcn}"     # slowest first
SAMPLERS="${SAMPLERS:-cosine md_K2_matched md_K2_top_n md_K1_matched md_K3_matched md_K5_matched}"
SEEDS="${SEEDS:-4}"; INITS="${INITS:-2}"
DEVICE="${DEVICE:-cpu}"; PAR="${PAR:-10}"   # container CPU quota is ~31 cores (cgroup cpu.max), NOT nproc=128; Gemma workers need ~6
OUT="${OUT:-results/flag_md/$VARIANT}"
mkdir -p "$OUT" logs/flag_md

sampler_args() {
  case "$1" in
    cosine)          echo "--sampling-strategy semantic" ;;
    md_K*_matched)   k=${1#md_K}; echo "--sampling-strategy markov_diffusion --diffusion-steps ${k%_matched} --md-selection matched_cosine" ;;
    md_K*_top_n)     k=${1#md_K}; echo "--sampling-strategy markov_diffusion --diffusion-steps ${k%_top_n} --md-selection top_n" ;;
    *) echo "unknown sampler $1" >&2; return 1 ;;
  esac
}

run_job() {
  local ds=$1 model=$2 sampler=$3 dir="$OUT/$3"
  local have; have=$(ls "$dir" 2>/dev/null | grep -c "^${ds}__${model}__${VARIANT}__" || true)
  if [ "$have" -ge $((SEEDS*INITS)) ]; then echo "skip  $ds $model $sampler ($have runs present)"; return 0; fi
  # A partially finished job is redone from scratch: leaving its files would
  # duplicate (seed, init) pairs on restart.
  rm -f "$dir/${ds}__${model}__${VARIANT}__"*.json
  $PY -m scripts.train.run --dataset "$ds" --model "$model" --variant "$VARIANT" \
      $(sampler_args "$sampler") --seeds "$SEEDS" --inits "$INITS" --device "$DEVICE" \
      --results-dir "$dir" > "logs/flag_md/${VARIANT}_${ds}_${model}_${sampler}.log" 2>&1 \
    && echo "done  $ds $model $sampler" || echo "FAIL  $ds $model $sampler"
}
export -f run_job sampler_args
export PY VARIANT SEEDS INITS DEVICE OUT

for m in $MODELS; do for s in $SAMPLERS; do for d in $DATASETS; do echo "$d $m $s"; done; done; done \
  | xargs -P "$PAR" -L1 bash -c 'run_job "$@"' _
