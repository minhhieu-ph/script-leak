#!/usr/bin/env bash
# Create the Python environment for scriptleak.
#
#   bash scripts/setup_env.sh                 # conda env "scriptleak", PyTorch for CUDA 13
#   ENV_NAME=myenv TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/setup_env.sh
#   SKIP_TORCH=1 bash scripts/setup_env.sh    # keep the PyTorch already in the active env
#
# NOT TESTED where it was written (no network there). Each step prints what it
# does; if one fails, the rest can be run by hand.
set -euo pipefail

ENV_NAME="${ENV_NAME:-scriptleak}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# PyTorch wheel index. cu130 is what the NeMo Speech install page shows for
# CUDA 13.x; change it to match the driver (see https://pytorch.org/get-started).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
NEMO_REPO="${NEMO_REPO:-https://github.com/NVIDIA-NeMo/Speech.git}"
NEMO_DIR="${NEMO_DIR:-$PWD/third_party/NeMo}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== 1. conda environment: $ENV_NAME (python $PYTHON_VERSION)"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first, or create a Python $PYTHON_VERSION venv and rerun with SKIP_CONDA=1."
  [ "${SKIP_CONDA:-0}" = "1" ] || exit 1
fi
if [ "${SKIP_CONDA:-0}" != "1" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
  conda activate "$ENV_NAME"
fi
python --version

echo "== 2. PyTorch"
if [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo "skipped (SKIP_TORCH=1)"
else
  pip install torch torchaudio --index-url "$TORCH_INDEX"
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
    x = torch.randn(1024, 1024, device="cuda")
    print("gpu matmul ok:", float((x @ x).abs().mean()) > 0)
else:
    print("WARNING: PyTorch cannot see a GPU. Pick another TORCH_INDEX for this driver.")
PY

echo "== 3. NeMo (ASR collection) and the other packages"
pip install Cython packaging
pip install 'nemo-toolkit[asr]'
pip install numpy scipy soundfile sentencepiece datasets huggingface_hub pythainlp pytest

echo "== 4. NeMo source tree (only for its config files)"
if [ ! -d "$NEMO_DIR/.git" ]; then
  mkdir -p "$(dirname "$NEMO_DIR")"
  git clone "$NEMO_REPO" "$NEMO_DIR"
fi
NEMO_VERSION="$(python -c 'import nemo; print(nemo.__version__)')"
echo "installed nemo-toolkit: $NEMO_VERSION"
if git -C "$NEMO_DIR" rev-parse -q --verify "refs/tags/v$NEMO_VERSION" >/dev/null; then
  git -C "$NEMO_DIR" checkout -q "v$NEMO_VERSION"
  echo "checked out tag v$NEMO_VERSION"
else
  echo "WARNING: no tag v$NEMO_VERSION in $NEMO_DIR; configs come from the default branch"
fi
echo "FastConformer hybrid configs found:"
find "$NEMO_DIR" -path '*fastconformer*' -name '*hybrid*ctc*bpe*.yaml' | sed 's/^/  /' || true

echo "== 5. self-checks"
cd "$HERE"
python -m pytest -q
python -m scriptleak.nemo_tools.check_env || true

echo
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Use one of the config files listed in step 4 as --base-config for training."
