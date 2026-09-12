#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a Linux GPU instance.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export FLAG_DEVICE="cuda:0"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

echo "[1/14] Checking GPU"
nvidia-smi

echo "[2/14] Installing the pinned GPU environment"
VENV=.venv-gpu bash scripts/setup/install_gpu.sh

echo "[3/14] Activating the GPU environment"
# shellcheck disable=SC1091
source .venv-gpu/bin/activate

echo "[4/14] Pinning the compatible Transformers stack"
python -m pip install --force-reinstall --no-cache-dir \
  "transformers==4.44.2" \
  "sentence-transformers==3.0.1" \
  "peft==0.12.0"

echo "[5/14] Fetching pinned upstream repositories"
bash scripts/setup/fetch_methods.sh
bash scripts/setup/fetch_methods.sh --verify

echo "[6/14] Verifying CUDA and the runtime"
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
PY

echo "[7/14] Running the smoke test"
python -m scripts.smoke_test

echo "[8/14] Authenticating with Hugging Face"
if [ -z "${HF_TOKEN:-}" ]; then
  read -r -s -p "Paste your Hugging Face read token: " HF_TOKEN
  echo
fi
if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: no Hugging Face token was provided."
  exit 1
fi
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo "[9/14] Downloading GLBench datasets"
python -m scripts.download.glbench --dataset all

echo "[10/14] Building the 1:10 benchmark datasets"
python -m scripts.preprocess.build_benchmark --dataset all

echo "[11/14] Encoding benchmark text with Sentence-BERT on the GPU"
python -m scripts.preprocess.encode_text \
  --dataset all \
  --device cuda:0 \
  --batch-size 128

echo "[12/14] Building semantic FLAG subgraphs"
python -m scripts.preprocess.sample_subgraphs --dataset all

echo "[13/14] Building ordinary baseline subgraphs"
python -m scripts.preprocess.sample_subgraphs \
  --dataset all \
  --strategy none

echo "[14/14] Dry-running the LLM prompt and smoke-generating 20 Reddit subgraphs"
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

echo "[15/15] Generating the complete discriminative and residual LLM cache"
python -m scripts.llm.generate_text \
  --dataset all \
  --kind both \
  --device cuda:0

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

echo
echo "Completed dataset preparation and GPU LLM generation."
echo "The generated cache is in cache/llm/."
echo "Do not run the flag training variant until its cache-loading integration is available."