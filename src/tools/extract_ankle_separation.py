"""
Extract left–right ankle separation signals from keypoints_common.json.

For each frame we compute horizontal ankle separation:

  sep_px   = |x_right_ankle - x_left_ankle|
  sep_norm = sep_px / body_height

Body height is taken from bbox_h if available, otherwise eye_to_ankle.

We save a CSV with:

  frame_idx, ms,
  sep_px, sep_norm,
  conf_mean,
  eye_to_ankle, bbox_h

Usage example (from project root):

python src/tools/extract_ankle_separation_signal.py \
  --input tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam4/keypoints_common.json \
  --output tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam4/ankle_separation.csv
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


def extract_ankle_separation(
    common_data: Dict[str, Any],
    conf_threshold: float = 0.3,
):
    """
    Return list of dict rows with ankle separation per frame.
    """
    rows = []

    for f in common_data.get("frames", []):
        frame_idx = f.get("frame_idx")
        ms = f.get("ms")

        height_px = f.get("height_px", {}) or {}
        eye_to_ankle = height_px.get("eye_to_ankle")
        bbox_h = height_px.get("bbox_h")

        la = get_joint(f, "left_ankle")
        ra = get_joint(f, "right_ankle")

        sep_px = math.nan
        sep_norm = math.nan
        conf_mean = math.nan

        if la and ra:
            if la["conf"] >= conf_threshold and ra["conf"] >= conf_threshold:
                # horizontal separation in pixels (x-axis)
                sep_px = abs(ra["x"] - la["x"])
                conf_mean = 0.5 * (la["conf"] + ra["conf"])

                # body height for normalization (same logic as other tools)
                if bbox_h is not None and bbox_h > 0:
                    body_h = bbox_h
                elif eye_to_ankle is not None and eye_to_ankle > 0:
                    body_h = eye_to_ankle
                else:
                    body_h = None

                if body_h is not None and math.isfinite(sep_px):
                    sep_norm = sep_px / body_h

        rows.append(
            {
                "frame_idx": frame_idx,
                "ms": ms,
                "sep_px": sep_px,
                "sep_norm": sep_norm,
                "conf_mean": conf_mean,
                "eye_to_ankle": eye_to_ankle if eye_to_ankle is not None else "",
                "bbox_h": bbox_h if bbox_h is not None else "",
            }
        )

    return rows


def save_csv(rows, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame_idx",
        "ms",
        "sep_px",
        "sep_norm",
        "conf_mean",
        "eye_to_ankle",
        "bbox_h",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if not rows:
        print(f"[extract_ankle_separation_signal] WARNING: no rows written")
    else:
        print(f"[extract_ankle_separation_signal] saved {len(rows)} rows -> {out_path}")


def convert_file(in_path: str | Path, out_path: str | Path, conf_threshold: float):
    data = load_json(in_path)
    rows = extract_ankle_separation(data, conf_threshold=conf_threshold)
    save_csv(rows, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Extract left–right ankle separation signals from keypoints_common.json"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to keypoints_common.json",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output ankle_separation.csv",
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
