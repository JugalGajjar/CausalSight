#!/usr/bin/env bash
# Colab bootstrap. Run in a cell:  !bash /content/CausalSight/colab/setup.sh <frames_zip_on_drive>
# Assumes Drive is mounted at /content/drive and the repo is cloned at /content/CausalSight.
set -euo pipefail
FRAMES_ZIP="${1:?path to frames zip on Drive, e.g. /content/drive/MyDrive/causalsight/train.zip}"
cd /content/CausalSight
pip install -q -e . 
pip install -q "transformers>=4.51" "peft>=0.13" "qwen-vl-utils>=0.0.8" accelerate safetensors pyyaml av pillow
mkdir -p /content/data && cd /content/data
[ -d "$(basename "${FRAMES_ZIP%.zip}")" ] || unzip -q "$FRAMES_ZIP" -d /content/data
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "frames at /content/data/$(basename "${FRAMES_ZIP%.zip}")"
