#!/usr/bin/env bash
# CausalVQA (Foss et al., NeurIPS 2025 D&B). Released via facebookresearch/CausalVQA;
# follow that repo's access instructions for the video segments, then place them here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/causalvqa"
mkdir -p "$DEST" && cd "$DEST"
[ -d CausalVQA ] || git clone --depth 1 https://github.com/facebookresearch/CausalVQA.git
echo "Repo cloned to $DEST/CausalVQA. Follow its README for the video download."
