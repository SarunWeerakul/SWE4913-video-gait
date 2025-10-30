#!/usr/bin/env bash
set -e
BASE="UNB-HTL-1001/BF/W1"
METHODS=("method1-yolo" "method2-mediapipe" "method3-posepipe")
CAMS=(1 2 3 4 5 6 7)
for M in "${METHODS[@]}"; do
  for C in "${CAMS[@]}"; do
    mkdir -p "tmp/$BASE/$M/cam${C}/vis_${M#*-}"
  done
done
echo "✅ Created tmp structure."
