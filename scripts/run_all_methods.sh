#!/usr/bin/env bash
set -euo pipefail
# Usage:
#   scripts/run_all_methods.sh <WALK_DIR> [N_OVERLAY]
# Example:
#   scripts/run_all_methods.sh data/raw/UNB-HTL-1001/BF/W1 120

WALK_DIR="${1:?WALK_DIR required}"
N_OVERLAY="${2:-120}"

# YOLO-Pose
bash scripts/run_pose_walk.sh yolo "$WALK_DIR" "$N_OVERLAY"

# Top-down (HRNet default; set POSE_EST=rtmpose to switch)
POSE_EST="${POSE_EST:-hrnet}" \
bash scripts/run_pose_walk.sh topdown "$WALK_DIR" "$N_OVERLAY"

# MediaPipe Pose
bash scripts/run_pose_walk.sh posepipe "$WALK_DIR" "$N_OVERLAY"
