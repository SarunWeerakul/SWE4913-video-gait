# scripts/run_pose_walk_autodiscover.sh
#!/usr/bin/env bash
set -euo pipefail
METHOD="${1:?METHOD yolo|topdown|posepipe}"; WALK_DIR="${2:?walk dir}"
N_OVERLAY="${3:-150}"

SUBPATH="${WALK_DIR#data/raw/}"
OUT_BASE="tmp/${SUBPATH}"
case "$METHOD" in
  yolo) OUT_METHOD="method1_yolo" ;;
  topdown) OUT_METHOD="method2_topdown" ;;
  posepipe) OUT_METHOD="method3_posepipe" ;;
  *) echo "unknown method"; exit 2 ;;
esac

mkdir -p "logs" "${OUT_BASE}/${OUT_METHOD}"

shopt -s nullglob
for IN_MP4 in "${WALK_DIR}"/*.mp4; do
  BN="$(basename "$IN_MP4")"                # camera_3.mp4
  CAM="${BN%.*}"; CAM="${CAM#camera_}"      # 3
  OUT_DIR="${OUT_BASE}/${OUT_METHOD}/cam${CAM}"
  OUT_JSON="${OUT_DIR}/keypoints.json"
  VIS_DIR="${OUT_DIR}/vis"
  OUT_MP4="${OUT_DIR}/overlay.mp4"
  mkdir -p "$OUT_DIR"
  echo "[RUN] $BN → cam${CAM}"
  PYTHONPATH=src python -m pose.cli --method "$METHOD" \
    "$IN_MP4" "$OUT_JSON" "$VIS_DIR" "$N_OVERLAY" "$OUT_MP4" \
    2>&1 | tee "logs/${OUT_METHOD}_cam${CAM}.log"
done
echo "[DONE] ${OUT_BASE}/${OUT_METHOD}"
