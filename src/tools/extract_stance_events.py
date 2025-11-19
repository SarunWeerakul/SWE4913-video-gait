"""
Detect foot stance events (left/right ankle) from keypoints_common.json.

We assume the common format produced by build_common_format.py:

{
  "schema_version": "pose_common_v1",
  "frames": [
    {
      "frame_idx": ...,
      "ms": ...,
      "joints": {
        "left_ankle": {"x": ..., "y": ..., "conf": ...},
        "right_ankle": {"x": ..., "y": ..., "conf": ...},
        ...
      }
    },
    ...
  ]
}

For each foot (left_ankle, right_ankle) we:

  1) Extract x, y, conf, ms over time.
  2) Compute per-frame "speed" = sqrt(dx^2 + dy^2).
  3) Mark frames where speed < still_threshold_px and conf > conf_threshold
     as stance-candidate frames.
  4) Find contiguous runs of stance-candidate frames with length >= min_stance_frames.
  5) For each run, create one stance event with:
        - foot ("left" or "right")
        - stance_id (1,2,3,... per foot)
        - frame_idx (center frame of the run)
        - ms (time at center frame)
        - x_ref, y_ref (mean position over the run)
        - duration_frames (length of the run)

Output CSV:

foot,stance_id,frame_idx,ms,x_ref,y_ref,duration_frames

Usage example (from project root):

python src/tools/extract_stance_events.py \
  --input tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/keypoints_common.json \
  --output tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/stance_events.csv

You can tweak parameters with:

  --still-threshold 2.0 \
  --conf-threshold 0.3 \
  --min-stance-frames 8
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FootSample:
    frame_idx: int
    ms: float
    x: float
    y: float
    conf: float


@dataclass
class StanceEvent:
    foot: str          # "left" or "right"
    stance_id: int
    frame_idx: int
    ms: float
    x_ref: float
    y_ref: float
    duration_frames: int


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def extract_foot_samples(
    common_data: Dict[str, Any],
    joint_name: str,
) -> List[FootSample]:
    """
    Extract a time series of (frame_idx, ms, x, y, conf) for a given ankle joint.
    joint_name: "left_ankle" or "right_ankle"
    """
    frames = common_data.get("frames", [])
    samples: List[FootSample] = []

    for f in frames:
        frame_idx = f.get("frame_idx")
        ms = f.get("ms")
        joints = f.get("joints", {})
        j = joints.get(joint_name)
        if j is None:
            continue
        if any(k not in j for k in ("x", "y", "conf")):
            continue

        x = float(j["x"])
        y = float(j["y"])
        conf = float(j["conf"])

        samples.append(FootSample(frame_idx=frame_idx, ms=ms, x=x, y=y, conf=conf))

    return samples


def detect_stance_events_for_foot(
    samples: List[FootSample],
    foot_label: str,
    still_threshold_px: float,
    conf_threshold: float,
    min_stance_frames: int,
) -> List[StanceEvent]:
    """
    Detect stance events for one foot using simple position-change thresholding.

    - samples: ordered by time (frame_idx increasing)
    - still_threshold_px: maximum per-frame movement (in px) to consider "still"
    - conf_threshold: minimum confidence for a sample to be used
    - min_stance_frames: minimum length of contiguous "still" frames to form a stance event
    """
    if len(samples) < 2:
        return []

    # compute per-frame speed
    n = len(samples)
    speeds: List[float] = [0.0] * n
    for i in range(1, n):
        dx = samples[i].x - samples[i - 1].x
        dy = samples[i].y - samples[i - 1].y
        speeds[i] = math.sqrt(dx * dx + dy * dy)

    # stance-candidate mask
    stance_candidate: List[bool] = [False] * n
    for i in range(n):
        s = speeds[i]
        c = samples[i].conf
        if c >= conf_threshold and s <= still_threshold_px:
            stance_candidate[i] = True

    # find contiguous runs of True
    events: List[StanceEvent] = []
    i = 0
    stance_id = 1

    while i < n:
        if not stance_candidate[i]:
            i += 1
            continue

        # start of run
        start = i
        while i < n and stance_candidate[i]:
            i += 1
        end = i  # stance_candidate[start:end] are True, end is first False or n

        length = end - start
        if length < min_stance_frames:
            # discard short run
            continue

        # create stance event
        seg = samples[start:end]
        # center frame index in the segment
        center_idx = start + length // 2
        center_sample = samples[center_idx]

        # average position
        x_mean = sum(s.x for s in seg) / len(seg)
        y_mean = sum(s.y for s in seg) / len(seg)

        event = StanceEvent(
            foot=foot_label,
            stance_id=stance_id,
            frame_idx=center_sample.frame_idx,
            ms=center_sample.ms,
            x_ref=x_mean,
            y_ref=y_mean,
            duration_frames=length,
        )
        events.append(event)
        stance_id += 1

    return events


def extract_stance_events(
    common_data: Dict[str, Any],
    still_threshold_px: float,
    conf_threshold: float,
    min_stance_frames: int,
) -> List[StanceEvent]:
    """
    Detect stance events for both left and right ankles.
    """
    events_all: List[StanceEvent] = []

    for joint_name, foot_label in (("left_ankle", "left"), ("right_ankle", "right")):
        samples = extract_foot_samples(common_data, joint_name=joint_name)
        if not samples:
            continue

        # sort by frame_idx (just in case)
        samples = sorted(samples, key=lambda s: s.frame_idx)

        ev = detect_stance_events_for_foot(
            samples,
            foot_label=foot_label,
            still_threshold_px=still_threshold_px,
            conf_threshold=conf_threshold,
            min_stance_frames=min_stance_frames,
        )
        events_all.extend(ev)

    # sort all events by time
    events_all.sort(key=lambda e: e.ms)
    return events_all


def save_events_csv(events: List[StanceEvent], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["foot", "stance_id", "frame_idx", "ms", "x_ref", "y_ref", "duration_frames"]
        )
        for e in events:
            writer.writerow(
                [
                    e.foot,
                    e.stance_id,
                    e.frame_idx,
                    f"{e.ms:.3f}",
                    f"{e.x_ref:.3f}",
                    f"{e.y_ref:.3f}",
                    e.duration_frames,
                ]
            )

    print(f"[extract_stance_events] saved -> {out_path}")


def convert_file(
    in_path: str | Path,
    out_path: str | Path,
    still_threshold_px: float,
    conf_threshold: float,
    min_stance_frames: int,
) -> None:
    data = load_json(in_path)
    events = extract_stance_events(
        data,
        still_threshold_px=still_threshold_px,
        conf_threshold=conf_threshold,
        min_stance_frames=min_stance_frames,
    )
    save_events_csv(events, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Detect foot stance events from keypoints_common.json"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to keypoints_common.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output stance_events.csv",
    )
    parser.add_argument(
        "--still-threshold",
        type=float,
        default=2.0,
        help="Max per-frame movement (px) to still be considered 'stance'. Default: 2.0",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Min joint confidence to use the sample. Default: 0.3",
    )
    parser.add_argument(
        "--min-stance-frames",
        type=int,
        default=8,
        help="Minimum length (frames) of a stance segment. Default: 8",
    )

    args = parser.parse_args()

    convert_file(
        args.input,
        args.output,
        still_threshold_px=args.still_threshold,
        conf_threshold=args.conf_threshold,
        min_stance_frames=args.min_stance_frames,
    )


if __name__ == "__main__":
    main()
