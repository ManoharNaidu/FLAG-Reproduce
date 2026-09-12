#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a Linux GPU instance.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

START_STEP=1
while [ "$#" -gt 0 ]; do
  case "$1" in
    --from-step)
      [ "$#" -ge 2 ] || { echo "ERROR: --from-step needs a number"; exit 2; }
      START_STEP="$2"
      shift 2
      ;;
    --from-step=*)
      START_STEP="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--from-step N]"
      echo "  --from-step N  resume at step N (1-15); completed earlier steps are skipped"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 [--from-step N]"
      exit 2
      ;;
  esac
done

if ! [[ "$START_STEP" =~ ^[0-9]+$ ]] || [ "$START_STEP" -lt 1 ] || [ "$START_STEP" -gt 15 ]; then
  echo "ERROR: --from-step must be an integer from 1 to 15"
  exit 2
fi

should_run() {
  [ "$START_STEP" -le "$1" ]
}

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export FLAG_DEVICE="cuda:0"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
# GLBench and this project's cached PyG graphs are trusted pickled Data objects.
# PyTorch 2.6+ otherwise defaults torch.load() to weights_only=True and rejects
# those files before preprocessing can inspect them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if should_run 1; then
  echo "[1/15] Checking GPU"
  nvidia-smi
fi

if should_run 2; then
  echo "[2/15] Installing the pinned GPU environment"
  VENV=.venv-gpu bash scripts/setup/install_gpu.sh
fi

if [ ! -x .venv-gpu/bin/python ]; then
  echo "ERROR: .venv-gpu is missing. Run without --from-step first."
  exit 1
fi
# shellcheck disable=SC1091
source .venv-gpu/bin/activate
echo "[3/15] GPU environment activated"

if should_run 4; then
  echo "[4/15] Pinning the compatible Transformers stack"
  python -m pip install --force-reinstall --no-cache-dir \
    "transformers==4.44.2" \
    "sentence-transformers==3.0.1" \
    "peft==0.12.0"
fi

if should_run 5; then
  echo "[5/15] Fetching pinned upstream repositories"
  bash scripts/setup/fetch_methods.sh
  bash scripts/setup/fetch_methods.sh --verify
fi

if should_run 6; then
  echo "[6/15] Verifying CUDA and the runtime"
  python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PY
fi

if should_run 7; then
  echo "[7/15] Running the smoke test"
  python -m scripts.smoke_test
fi

if should_run 8; then
  echo "[8/15] Authenticating with Hugging Face"
  if [ -z "${HF_TOKEN:-}" ]; then
    read -r -s -p "Paste your Hugging Face read token: " HF_TOKEN
    echo
  fi
  if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: no Hugging Face token was provided."
    exit 1
  fi
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
fi

if should_run 9; then
  echo "[9/15] Downloading GLBench datasets"
  python -m scripts.download.glbench --dataset all
fi

if should_run 10; then
  echo "[10/15] Building the 1:10 benchmark datasets"
  python -m scripts.preprocess.build_benchmark --dataset all
fi

if should_run 11; then
  echo "[11/15] Encoding benchmark text with Sentence-BERT on the GPU"
  python -m scripts.preprocess.encode_text \
    --dataset all \
    --device cuda:0 \
    --batch-size 128
fi

if should_run 12; then
  echo "[12/15] Building semantic FLAG subgraphs"
  python -m scripts.preprocess.sample_subgraphs --dataset all
fi

if should_run 13; then
  echo "[13/15] Building ordinary baseline subgraphs"
  python -m scripts.preprocess.sample_subgraphs \
    --dataset all \
    --strategy none
fi

if should_run 14; then
  echo "[14/15] Dry-running the LLM prompt and smoke-generating 20 Reddit subgraphs"
  python -m scripts.llm.generate_text \
    --dataset reddit \
    --kind discriminative \
    --device cuda:0 \
    --dry-run
  python -m scripts.llm.generate_text \
    --dataset reddit \
    --kind discriminative \
    --device cuda:0 \
    --limit 20
fi

if should_run 15; then
  echo "[15/15] Generating complete LLM caches in parallel across visible GPUs"
  NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')}" \
  GENERATE_ARGS="${GENERATE_ARGS:---truncate-chars 300 --max-new-tokens 64 --force}" \
    bash scripts/setup/run_multi_gpu_llm.sh

  echo "Verifying generated LLM cache checksums"
  python - <<'PY'
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
  fi

echo
echo "Completed dataset preparation and GPU LLM generation."
echo "The generated cache is in cache/llm/."
echo "Do not run the flag training variant until its cache-loading integration is available."