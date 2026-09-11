#!/usr/bin/env bash
# Downloads CLEVRER into data/raw/clevrer/. URLs verified against http://clevrer.csail.mit.edu/ on 2026-09-10.
#   videos/            video_train.zip, video_validation.zip
#   annotations/       annotation_train.zip, annotation_validation.zip  (3D trajectories, collisions, properties)
#   questions/         train.json, validation.json                      (programs, choices)
#   derender_proposals.zip                                              (per-frame 2D object masks; source of evidence boxes)
# Sizes: videos are the bulk (tens of GB).
#   --no-videos            annotations, questions, proposals only
#   --videos validation    add validation videos only (enough for Stage 0 evaluation)
#   (default)              everything
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/clevrer"
BASE="http://data.csail.mit.edu/clevrer"
WANT_VIDEOS="train validation"
case "${1:-}" in
  --no-videos) WANT_VIDEOS="" ;;
  --videos) WANT_VIDEOS="${2:?usage: --videos train|validation}" ;;
esac
mkdir -p "$DEST" && cd "$DEST"

fetch() {  # fetch <relative-path>
  mkdir -p "$(dirname "$1")"
  if [ -f "$1" ]; then echo "have $1"; else echo "get  $1"; curl -L --fail --progress-bar -o "$1" "$BASE/$1"; fi
}

fetch README.txt
fetch questions/train.json
fetch questions/validation.json
fetch annotations/train/annotation_train.zip
fetch annotations/validation/annotation_validation.zip
fetch derender_proposals.zip
for sp in $WANT_VIDEOS; do
  fetch "videos/$sp/video_$sp.zip"
done

for z in $(find . -name '*.zip'); do
  d="${z%.zip}"
  if [ ! -d "$d" ]; then echo "unzip $z"; unzip -q "$z" -d "$d"; fi
done
echo "CLEVRER ready in $DEST"
