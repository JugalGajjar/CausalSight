#!/usr/bin/env bash
# Downloads CLEVRER into data/raw/clevrer/. URLs verified against http://clevrer.csail.mit.edu/ on 2026-09-10.
#   videos/            video_train.zip, video_validation.zip
#   annotations/       annotation_train.zip, annotation_validation.zip  (3D trajectories, collisions, properties)
#   questions/         train.json, validation.json                      (programs, choices)
#   derender_proposals.zip                                              (per-frame 2D object masks; source of evidence boxes)
# Sizes: videos are the bulk (tens of GB). Pass --no-videos to skip them for annotation-only work.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/clevrer"
BASE="http://data.csail.mit.edu/clevrer"
WANT_VIDEOS=1
[ "${1:-}" = "--no-videos" ] && WANT_VIDEOS=0
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
if [ "$WANT_VIDEOS" = 1 ]; then
  fetch videos/train/video_train.zip
  fetch videos/validation/video_validation.zip
fi

for z in $(find . -name '*.zip'); do
  d="${z%.zip}"
  if [ ! -d "$d" ]; then echo "unzip $z"; unzip -q "$z" -d "$d"; fi
done
echo "CLEVRER ready in $DEST"
