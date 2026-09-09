#!/usr/bin/env bash
# Bootstrap a GPU instance (vast.ai or similar) for the LLM stage.
#
#   bash scripts/setup/install_gpu.sh
#
# Installs the CUDA stack, verifies the GPU is actually usable, and re-runs the
# numerical-correctness checks. See docs/vastai_gpu_workflow.md.
#
# On CPU-only machines this exits with guidance rather than installing a broken
# environment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv-gpu}"

# CUDA-matched pins. The CPU pins in environment/cpu.lock.txt exist because
# torch 2.4.0+cpu returns SILENTLY WRONG numbers on the reference machine
# (research/compatibility_notes.md section 3). That is a CPU-wheel finding and
# does not necessarily apply to CUDA builds, so we pin conservatively to the
# same minor version and re-verify numerics below rather than assuming.
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu121}"
PYG_VERSION="${PYG_VERSION:-2.3.1}"

echo "=============================================================="
echo "FLAG benchmark - GPU bootstrap"
echo "=============================================================="
echo "  root        $ROOT"
echo "  python      $($PYTHON --version 2>&1)"
echo "  venv        $VENV"
echo "  torch       ${TORCH_VERSION}+${CUDA_CHANNEL}"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found -- this does not look like a GPU machine."
  echo
  echo "  The LLM stage is GPU-only by decision D-003: gemma-2-9b-it is about"
  echo "  18.5 GB in fp16 and no smaller substitute is provided, because its"
  echo "  numbers would not be comparable to the paper's."
  echo
  echo "  Everything else runs on CPU. For that, use:"
  echo "      python -m venv .venv-cpu"
  echo "      .venv-cpu/bin/pip install -r environment/cpu.lock.txt"
  exit 1
fi

echo "--- detected GPUs ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader || true
echo

VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
           | head -1 | tr -d ' ')"
if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" -lt 23000 ]; then
  echo "WARNING: ${VRAM_MB} MiB of VRAM detected."
  echo "  gemma-2-9b-it in fp16 needs ~18.5 GB of weights plus a KV cache."
  echo "  24 GB or more is recommended; expect OOM below that."
  echo
fi

echo "--- creating $VENV ---"
$PYTHON -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
[ -x "$PIP" ] || { PIP="$VENV/Scripts/pip.exe"; PY="$VENV/Scripts/python.exe"; }

"$PIP" install --quiet --upgrade pip

echo "--- torch ${TORCH_VERSION}+${CUDA_CHANNEL} ---"
"$PIP" install --quiet "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

echo "--- scientific stack ---"
"$PIP" install --quiet "numpy<2" scipy scikit-learn sympy pandas matplotlib \
    pyyaml tqdm gdown

echo "--- torch_geometric ${PYG_VERSION} ---"
# Pinned to 2.3.1: >= 2.4 crashed in SAGEConv on the reference machine, and
# FLAG's GAT uses SAGEConv as its second layer. PMP's official requirements pin
# the same version. torch_scatter is deliberately NOT installed -- the wheel
# destabilised PyG; src/flagbench/compat provides an equivalence-tested shim.
"$PIP" install --quiet "torch_geometric==${PYG_VERSION}"

echo "--- LLM stack ---"
"$PIP" install --quiet "transformers>=4.42" "sentence-transformers" \
    "peft" "accelerate"

echo "--- flagbench (editable) ---"
"$PIP" install --quiet -e .

echo
echo "--- verifying CUDA is usable ---"
"$PY" - <<'PY'
import sys
import torch

print(f"  torch            {torch.__version__}")
print(f"  cuda available   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("\nERROR: torch cannot see a GPU despite nvidia-smi working.")
    print("  Usually a driver/CUDA-runtime mismatch. Try a different")
    print("  CUDA_CHANNEL (cu118 / cu121 / cu124).")
    sys.exit(1)

print(f"  cuda version     {torch.version.cuda}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  device {i}         {props.name}, "
          f"{props.total_memory / 1024**3:.1f} GiB, sm_{props.major}{props.minor}")

# A real allocation, not just a capability query.
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = (x @ x).float()
assert torch.isfinite(y).all(), "fp16 matmul produced non-finite values"
print("  fp16 matmul      OK")
PY

echo
echo "--- numerical correctness (guards against a silently-wrong build) ---"
"$PY" -m scripts.smoke_test || {
  echo
  echo "SMOKE TEST FAILED. Do not run experiments on this environment."
  exit 1
}

"$PY" -m pip freeze > environment/gpu.lock.txt
echo
echo "=============================================================="
echo "ready. wrote environment/gpu.lock.txt"
echo "=============================================================="
cat <<'NEXT'

Next:

  1. Accept the Gemma licence and authenticate (the repo is gated):
       https://huggingface.co/google/gemma-2-9b-it
       huggingface-cli login --token "$HF_TOKEN"

  2. Copy your prepared inputs up (do NOT rebuild the benchmark here -- it is
     built from your seed, and a rebuild would give a different graph):
       data/benchmark/  cache/sampling/  prompts/

  3. Smoke-test before committing to the full run:
       python -m scripts.llm.generate_text --dataset reddit \
              --kind discriminative --limit 20
     Check the reported node coverage.

  4. Full run, then copy cache/llm/ back and verify the checksums:
       python -m scripts.llm.generate_text --dataset all --kind both

Full workflow: docs/vastai_gpu_workflow.md
NEXT
