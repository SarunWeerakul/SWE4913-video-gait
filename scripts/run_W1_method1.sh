#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
POSE="python -m pose.cli"

BASE="UNB-HTL-1001/BF/W1"
RAW="$PWD/data/raw/$BASE"
OUT="$PWD/tmp/$BASE"
CAMS=(1 2 3 4 5 6 7)

for CAM in "${CAMS[@]}"; do
  IN="$RAW/camera_${CAM}.mp4"
  D="$OUT/method1_yolo/cam${CAM}"
  mkdir -p "$D/vis"
  $POSE "$IN" "$D/pose.json" "$D/vis" 6 "$D/overlay.mp4" --method method1_yolo
  echo "✅ method1_yolo cam${CAM}"
done
