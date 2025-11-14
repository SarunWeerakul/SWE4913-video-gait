#!/usr/bin/env bash
set -euo pipefail
# Pre-create tmp tree for a given walk
# Usage: scripts/setup_tmp.sh data/raw/UNB-HTL-1001/BF/W1
WALK_DIR="${1:?WALK_DIR required}"
SUBPATH="${WALK_DIR#data/raw/}"
for m in method1_yolo method2_topdown method3_posepipe; do
  mkdir -p "tmp/${SUBPATH}/${m}/vis"
done
echo "[OK] Created tmp/${SUBPATH}/{method*/vis}"
