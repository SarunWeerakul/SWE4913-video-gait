#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/run_pose_walk_robust.sh --method yolo --walk data/raw/UNB-HTL-1001/BF/W1 \
#       [--overlay 150] [--delay 0] [--resume] [--cams 1,2,3,4,5,6,7]
#
# Env knobs (optional):
#   POSE_EST=hrnet|rtmpose   # for --method topdown (defaults hrnet)
#   USE_TRACKER=1            # enable tracking in topdown pipeline

METHOD=""
WALK_DIR=""
N_OVERLAY=150
DELAY=0
RESUME=0
CAMS="1,2,3,4,5,6,7"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method)   METHOD="${2}"; shift 2 ;;
    --walk)     WALK_DIR="${2}"; shift 2 ;;
    --overlay)  N_OVERLAY="${2}"; shift 2 ;;
    --delay)    DELAY="${2}"; shift 2 ;;
    --resume)   RESUME=1; shift ;;
    --cams)     CAMS="${2}"; shift 2 ;;
    -h|--help)  sed -n '1,80p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

[[ -n "$METHOD"   ]] || { echo "--method required"; exit 2; }
[[ -n "$WALK_DIR" ]] || { echo "--walk required"; exit 2; }

SUBPATH="${WALK_DIR#data/raw/}"
OUT_BASE="tmp/${SUBPATH}"

case "$METHOD" in
  yolo)     OUT_METHOD="method1_yolo" ;;
  topdown)  OUT_METHOD="method2_topdown" ;;
  posepipe) OUT_METHOD="method3_posepipe" ;;
  *) echo "Unknown method: $METHOD"; exit 2 ;;
esac

mkdir -p "logs" "${OUT_BASE}/${OUT_METHOD}"
echo "[INFO] method=$METHOD walk=$WALK_DIR overlay=$N_OVERLAY delay=$DELAY resume=$RESUME cams=$CAMS"

# Quick model sanity hints (not fatal)
if [[ "$METHOD" == "yolo" ]]; then
  [[ -f "models/yolov8n-pose.pt" ]] || echo "[WARN] models/yolov8n-pose.pt not found"
elif [[ "$METHOD" == "topdown" ]]; then
  [[ -f "models/yolov8n.pt" ]] || echo "[WARN] models/yolov8n.pt not found"
elif [[ "$METHOD" == "posepipe" ]]; then
  [[ -f "models/pose_landmarker_full.task" ]] || echo "[WARN] models/pose_landmarker_full.task not found"
fi

IFS=',' read -r -a CAM_ARR <<< "$CAMS"
for cam in "${CAM_ARR[@]}"; do
  IN_MP4="${WALK_DIR}/camera_${cam}.mp4"
  OUT_JSON="${OUT_BASE}/${OUT_METHOD}/camera_${cam}.json"
  VIS_DIR="${OUT_BASE}/${OUT_METHOD}/vis/camera_${cam}"
  OUT_MP4="${OUT_BASE}/${OUT_METHOD}/camera_${cam}_overlay.mp4"

  if [[ ! -f "$IN_MP4" ]]; then
    echo "[WARN] Missing $IN_MP4 — skipping cam ${cam}."
    continue
  fi

  if [[ $RESUME -eq 1 && -s "$OUT_JSON" && -s "$OUT_MP4" ]]; then
    echo "[SKIP] cam ${cam} already has JSON+MP4 (resume on)"; continue
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

  if [[ "$DELAY" -gt 0 ]]; then
    echo "[SLEEP] ${DELAY}s"; sleep "$DELAY"
  fi
done

echo "[DONE] Outputs → ${OUT_BASE}/${OUT_METHOD}"
