"""
Extract upper-leg (thigh) angle signals from keypoints_common.json.

For each frame we compute:

  Left thigh vector  L = knee_L - hip_L
  Right thigh vector R = knee_R - hip_R

Then:

  angle_L = atan2(Ly, Lx)  # radians
  angle_R = atan2(Ry, Rx)  # radians
  angle_diff = wrap_to_pi(angle_L - angle_R)

We save a CSV with:

  frame_idx, ms,
  angle_L_rad, angle_R_rad, angle_diff_rad,
  angle_L_deg, angle_R_deg, angle_diff_deg

Usage example (from project root):

python src/tools/extract_thigh_angle_signal.py \
  --input tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/keypoints_common.json \
  --output tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/thigh_angle.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def get_joint(frame: Dict[str, Any], name: str) -> Optional[Dict[str, float]]:
    j = frame.get("joints", {}).get(name)
    if j is None:
        return None
    if not all(k in j for k in ("x", "y", "conf")):
        return None
    return {
        "x": float(j["x"]),
        "y": float(j["y"]),
        "conf": float(j["conf"]),
    }


def wrap_to_pi(angle: float) -> float:
    """
    Wrap angle in radians to [-pi, pi].
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def extract_thigh_angles(
    common_data: Dict[str, Any],
    conf_threshold: float = 0.3,
):
    """
    Return list of dict rows with thigh angles per frame.
    """
    rows = []

    for f in common_data.get("frames", []):
        frame_idx = f.get("frame_idx")
        ms = f.get("ms")

        lh = get_joint(f, "left_hip")
        rh = get_joint(f, "right_hip")
        lk = get_joint(f, "left_knee")
        rk = get_joint(f, "right_knee")

        angle_L_rad = math.nan
        angle_R_rad = math.nan
        angle_diff_rad = math.nan

        if lh and lk and rh and rk:
            # Check confidence
            if (
                lh["conf"] >= conf_threshold
                and lk["conf"] >= conf_threshold
                and rh["conf"] >= conf_threshold
                and rk["conf"] >= conf_threshold
            ):
                # Left thigh vector: hip -> knee
                Lx = lk["x"] - lh["x"]
                Ly = lk["y"] - lh["y"]

                # Right thigh vector: hip -> knee
                Rx = rk["x"] - rh["x"]
                Ry = rk["y"] - rh["y"]

                # Avoid zero-length vectors
                if abs(Lx) + abs(Ly) > 1e-6:
                    angle_L_rad = math.atan2(Ly, Lx)
                if abs(Rx) + abs(Ry) > 1e-6:
                    angle_R_rad = math.atan2(Ry, Rx)

                if not math.isnan(angle_L_rad) and not math.isnan(angle_R_rad):
                    angle_diff_rad = wrap_to_pi(angle_L_rad - angle_R_rad)

        # convert to degrees for easier interpretation/plotting
        angle_L_deg = angle_L_rad * 180.0 / math.pi if not math.isnan(angle_L_rad) else math.nan
        angle_R_deg = angle_R_rad * 180.0 / math.pi if not math.isnan(angle_R_rad) else math.nan
        angle_diff_deg = (
            angle_diff_rad * 180.0 / math.pi if not math.isnan(angle_diff_rad) else math.nan
        )

        rows.append(
            {
                "frame_idx": frame_idx,
                "ms": ms,
                "angle_L_rad": angle_L_rad,
                "angle_R_rad": angle_R_rad,
                "angle_diff_rad": angle_diff_rad,
                "angle_L_deg": angle_L_deg,
                "angle_R_deg": angle_R_deg,
                "angle_diff_deg": angle_diff_deg,
            }
        )

    return rows


def save_csv(rows, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_idx",
                    "ms",
                    "angle_L_rad",
                    "angle_R_rad",
                    "angle_diff_rad",
                    "angle_L_deg",
                    "angle_R_deg",
                    "angle_diff_deg",
                ]
            )
        return

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def convert_file(in_path: str | Path, out_path: str | Path, conf_threshold: float):
    data = load_json(in_path)
    rows = extract_thigh_angles(data, conf_threshold=conf_threshold)
    save_csv(rows, out_path)
    print(f"[extract_thigh_angle_signal] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract upper-leg (thigh) angle signals from keypoints_common.json"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to keypoints_common.json",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output thigh_angle.csv",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Minimum joint confidence to use (default=0.3)",
    )

    args = parser.parse_args()
    convert_file(args.input, args.output, conf_threshold=args.conf_threshold)


if __name__ == "__main__":
    main()
