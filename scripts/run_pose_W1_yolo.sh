#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:${PYTHONPATH:-}"
POSE_CMD="python -m pose.cli"

BASE="UNB-HTL-1001/BF/W1"
RAW="$PWD/data/raw/$BASE"
OUT="$PWD/tmp/$BASE"
CAMS=(1 2 3 4 5 6 7)

for CAM in "${CAMS[@]}"; do
  IN="$RAW/camera_${CAM}.mp4"
  D="$OUT/yolo/cam${CAM}"
  mkdir -p "$D/vis_yolo"

  POSE_JSON="$D/pose__yolo__baseline.json"
  OVERLAY="$D/pose_overlay_yolo__baseline.mp4"

  $POSE_CMD "$IN" "$POSE_JSON" "$D/vis_yolo" 6 "$OVERLAY" --method yolo

  cat > "$D/run_info.yaml" <<YAML
dataset: UNB-HTL-1001
subject: BF
walk: W1
camera: ${CAM}
pose:
  method: yolo
  params: { preset: baseline }
  out_json: $(basename "$POSE_JSON")
  vis_dir: vis_yolo
  overlay_mp4: $(basename "$OVERLAY")
YAML
  echo "✅ yolo cam${CAM}"
done
echo "✅ Done W1 yolo."
