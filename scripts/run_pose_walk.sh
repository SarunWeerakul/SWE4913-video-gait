#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_pose_walk.sh <METHOD> <WALK_DIR> [N_OVERLAY]
# Example:
#   scripts/run_pose_walk.sh yolo data/raw/UNB-HTL-1001/BF/W1 150
#
# METHOD: yolo | topdown | posepipe
# WALK_DIR: directory containing camera_1.mp4 ... camera_7.mp4
# N_OVERLAY: optional (default 150)

METHOD="${1:?METHOD required (yolo|topdown|posepipe)}"
WALK_DIR="${2:?WALK_DIR required (e.g., data/raw/UNB-HTL-1001/BF/W1)}"
N_OVERLAY="${3:-150}"

SUBPATH="${WALK_DIR#data/raw/}"                      # e.g., UNB-HTL-1001/BF/W1
OUT_BASE="tmp/${SUBPATH}"

case "$METHOD" in
  yolo)     OUT_METHOD="method1_yolo" ;;
  topdown)  OUT_METHOD="method2_topdown" ;;
  posepipe) OUT_METHOD="method3_posepipe" ;;
  *) echo "Unknown method: $METHOD"; exit 2 ;;
esac

mkdir -p "logs" "${OUT_BASE}/${OUT_METHOD}"

echo "[INFO] Running method=$METHOD on $WALK_DIR"
for cam in {1..7}; do
  IN_MP4="${WALK_DIR}/camera_${cam}.mp4"
  OUT_JSON="${OUT_BASE}/${OUT_METHOD}/camera_${cam}.json"
  VIS_DIR="${OUT_BASE}/${OUT_METHOD}/vis/camera_${cam}"
  OUT_MP4="${OUT_BASE}/${OUT_METHOD}/camera_${cam}_overlay.mp4"

  if [[ ! -f "$IN_MP4" ]]; then
    echo "[WARN] Missing $IN_MP4 — skipping."
    continue
  fi

  echo "[RUN ] cam ${cam} → ${OUT_METHOD}"
  PYTHONPATH=src python -m pose.cli \
    --method "$METHOD" \
    "$IN_MP4" \
    "$OUT_JSON" \
    "$VIS_DIR" \
    "$N_OVERLAY" \
    "$OUT_MP4" \
    2>&1 | tee "logs/pose_${OUT_METHOD}_cam${cam}.log"
done

echo "[DONE] Outputs → ${OUT_BASE}/${OUT_METHOD}"
