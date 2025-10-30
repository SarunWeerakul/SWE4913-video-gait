#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"
POSE_CMD="python -m pose.cli"

BASE="UNB-HTL-1001/BF/W1"
RAW="data/raw/$BASE"
OUT="tmp/$BASE"
CAMS=(1 2 3 4 5 6 7)

for CAM in "${CAMS[@]}"; do
  IN="$RAW/camera_${CAM}.mp4"
  D="$OUT/method1-yolo/cam${CAM}/yolo"   # <-- immediate folder must be 'yolo'
  mkdir -p "$D/vis_yolo"

  POSE_JSON="$D/pose__yolo__baseline.json"
  OVERLAY="$D/pose_overlay_yolo__baseline.mp4"

  $POSE_CMD "$IN" "$POSE_JSON" "$D/vis_yolo" 6 "$OVERLAY" --method yolo

  cat > "$OUT/method1-yolo/cam${CAM}/run_info.yaml" <<YAML
dataset: UNB-HTL-1001
subject: BF
walk: W1
camera: ${CAM}
pose:
  method: yolo
  params: { preset: baseline }
  out_json: yolo/$(basename "$POSE_JSON")
  vis_dir: yolo/vis_yolo
  overlay_mp4: yolo/$(basename "$OVERLAY")
YAML
  echo "✅ method1-yolo cam${CAM}"
done
echo "✅ Done W1 method1-yolo."
