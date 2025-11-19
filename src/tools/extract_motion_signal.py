"""
Extract gait motion signals from keypoints_common.json.

Input: common-format JSON produced by build_common_format.py:
{
  "schema_version": "pose_common_v1",
  "frames": [
    {
      "frame_idx": ...,
      "ms": ...,
      "height_px": { "eye_to_ankle": ..., "bbox_h": ... },
      "bbox": {...} or null,
      "joints": {
        "left_ankle": {"x": ..., "y": ..., "conf": ...},
        "right_ankle": {"x": ..., "y": ..., "conf": ...},
        "left_hip": {"x": ..., "y": ..., "conf": ...},
        "right_hip": {"x": ..., "y": ..., "conf": ...},
        ...
      }
    },
    ...
  ]
}

We output a CSV with one row per frame:

frame_idx,ms,
ankle_y,hip_y,
ankle_rel,              # ankle_y - hip_y
ankle_rel_norm,         # (ankle_y - hip_y) / body_height
ankle_conf_mean,
hip_conf_mean,
eye_to_ankle,bbox_h

This ankle_rel_norm is what we'll use for synchronization.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def safe_get_joint(frame: Dict[str, Any], name: str) -> Optional[Dict[str, float]]:
    joints = frame.get("joints", {})
    j = joints.get(name)
    if j is None:
        return None
    if any(k not in j for k in ("x", "y", "conf")):
        return None
    return j


def extract_signals(common_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a per-frame record with:
      - frame_idx, ms
      - ankle_y        (mean of left/right ankle y)
      - hip_y          (mean of left/right hip y)
      - ankle_rel      (ankle_y - hip_y)
      - ankle_rel_norm ((ankle_y - hip_y) / body_height)
      - ankle_conf_mean, hip_conf_mean
      - eye_to_ankle, bbox_h
    """
    frames = common_data.get("frames", [])
    rows: List[Dict[str, Any]] = []

    for f in frames:
        frame_idx = f.get("frame_idx")
        ms = f.get("ms")

        height_px = f.get("height_px", {}) or {}
        eye_to_ankle = height_px.get("eye_to_ankle")
        bbox_h = height_px.get("bbox_h")

        # collect ankles
        ankles_y: List[float] = []
        ankles_conf: List[float] = []

        for name in ("left_ankle", "right_ankle"):
            j = safe_get_joint(f, name)
            if j is not None:
                ankles_y.append(j["y"])
                ankles_conf.append(j["conf"])

        if ankles_y:
            ankle_y = sum(ankles_y) / len(ankles_y)
            ankle_conf_mean = sum(ankles_conf) / len(ankles_conf)
        else:
            ankle_y = math.nan
            ankle_conf_mean = math.nan

        # collect hips
        hips_y: List[float] = []
        hips_conf: List[float] = []

        for name in ("left_hip", "right_hip"):
            j = safe_get_joint(f, name)
            if j is not None:
                hips_y.append(j["y"])
                hips_conf.append(j["conf"])

        if hips_y:
            hip_y = sum(hips_y) / len(hips_y)
            hip_conf_mean = sum(hips_conf) / len(hips_conf)
        else:
            hip_y = math.nan
            hip_conf_mean = math.nan

        # relative ankle position: ankle below hip
        if math.isfinite(ankle_y) and math.isfinite(hip_y):
            ankle_rel = ankle_y - hip_y
        else:
            ankle_rel = math.nan

        # choose body height reference
        if bbox_h is not None and bbox_h > 0:
            body_h = bbox_h
        elif eye_to_ankle is not None and eye_to_ankle > 0:
            body_h = eye_to_ankle
        else:
            body_h = None

        if body_h is not None and math.isfinite(ankle_rel):
            ankle_rel_norm = ankle_rel / body_h
        else:
            ankle_rel_norm = math.nan

        row = {
            "frame_idx": frame_idx,
            "ms": ms,
            "ankle_y": ankle_y,
            "hip_y": hip_y,
            "ankle_rel": ankle_rel,
            "ankle_rel_norm": ankle_rel_norm,
            "ankle_conf_mean": ankle_conf_mean,
            "hip_conf_mean": hip_conf_mean,
            "eye_to_ankle": eye_to_ankle if eye_to_ankle is not None else "",
            "bbox_h": bbox_h if bbox_h is not None else "",
        }
        rows.append(row)

    return rows


def save_csv(rows: List[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_idx", "ms",
                    "ankle_y", "hip_y",
                    "ankle_rel", "ankle_rel_norm",
                    "ankle_conf_mean", "hip_conf_mean",
                    "eye_to_ankle", "bbox_h",
                ]
            )
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def convert_file(in_path: str | Path, out_path: str | Path) -> None:
    data = load_json(in_path)
    rows = extract_signals(data)
    save_csv(rows, out_path)
    print(f"[extract_motion_signal] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract relative ankle motion signal from keypoints_common.json"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to keypoints_common.json"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output CSV"
    )

    args = parser.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
