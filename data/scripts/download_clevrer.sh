#!/usr/bin/env bash
# Downloads CLEVRER videos, annotations, and questions into data/raw/clevrer/.
# Verify the file list against http://clevrer.csail.mit.edu/ before first use;
# the archive names below follow the official release page.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/clevrer"
BASE="http://data.csail.mit.edu/clevrer"
mkdir -p "$DEST" && cd "$DEST"
for f in videos/video_train.zip videos/video_validation.zip \
         annotations/annotation_train.zip annotations/annotation_validation.zip \
         questions/train.json questions/validation.json; do
  mkdir -p "$(dirname "$f")"
  [ -f "$f" ] || curl -L --fail -o "$f" "$BASE/$f"
done
for z in videos/*.zip annotations/*.zip; do
  d="${z%.zip}"; [ -d "$d" ] || unzip -q "$z" -d "$d"
done
echo "CLEVRER ready in $DEST"
