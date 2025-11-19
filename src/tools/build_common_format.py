"""
Convert raw pose keypoints into a common format for analysis.

Supported layouts:
- coco17  (e.g. YOLO pose)
- mp33    (MediaPipe Pose, 33 joints)

Input JSON (example):

{
  "schema_version": "1.0",
  "layout": "coco17" or "mp33",
  "method": "yolo" or "topdown_mp33",
  "video": "...",
  "fps": ...,
  "width": ...,
  "height": ...,
  "frames": [
    {
      "frame_idx": 0,
      "ms": 0.0,
      "people": [
        {
          "kpts": [x0, y0, c0, x1, y1, c1, ...],
          "score": 0.88,
          "tid": null or int,
          "bbox": [x, y, w, h] or null or missing
        },
        ...
      ]
    },
    ...
  ],
  "meta": {...}
}

Output JSON (common format):

{
  "schema_version": "pose_common_v1",
  "method": "yolo" or "topdown_mp33",
  "camera": "cam1",
  "video": "...",
  "fps": ...,
  "width": ...,
  "height": ...,
  "joints": [
    "left_eye",
    "right_eye",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
  ],
  "person_id": "main",
  "frames": [
    {
      "frame_idx": 0,
      "ms": 0.0,
      "height_px": {
        "eye_to_ankle": 310.5,   # vertical distance (px), or null
        "bbox_h": 340.2          # bbox height (px), or null
      },
      "bbox": {
        "x": ..., "y": ..., "w": ..., "h": ...
      } or null,
      "joints": {
        "left_eye":   {"x": ..., "y": ..., "conf": ...},
        "right_eye":  {"x": ..., "y": ..., "conf": ...},
        "left_hip":   {"x": ..., "y": ..., "conf": ...},
        "right_hip":  {"x": ..., "y": ..., "conf": ...},
        "left_knee":  {"x": ..., "y": ..., "conf": ...},
        "right_knee": {"x": ..., "y": ..., "conf": ...},
        "left_ankle": {"x": ..., "y": ..., "conf": ...},
        "right_ankle":{"x": ..., "y": ..., "conf": ...}
      }
    },
    ...
  ]
}

Rules:
- each frame has at most ONE person (the “main” one)
- main person = tid with most frames (if any tid exists),
  otherwise the person with the largest bbox area (if bbox available),
  otherwise just the first person in the frame.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# --------- COCO17 keypoint layout ---------
# (Used by method1_yolo)

COCO17_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]


# --------- MP33 keypoint layout (MediaPipe Pose) ---------
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# 0–32 landmarks

MP33_NAMES = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]


# joints we keep in the common format
TARGET_JOINTS = [
    "left_eye", "right_eye",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


# --------- basic IO helpers ---------

def load_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


# --------- choose / filter to main person ---------

def _has_any_tid(data: dict) -> bool:
    """Return True if ANY person in ANY frame has a non-null tid."""
    for frame in data.get("frames", []):
        for person in frame.get("people", []):
            if person.get("tid") is not None:
                return True
    return False


def _select_main_tid(data: dict):
    """
    Count in how many frames each tid appears and return
    the tid with the maximum frame count.
    """
    counts: dict[int, int] = defaultdict(int)

    for frame in data.get("frames", []):
        tids_in_frame = set()
        for person in frame.get("people", []):
            tid = person.get("tid")
            if tid is not None:
                tids_in_frame.add(tid)
        for tid in tids_in_frame:
            counts[tid] += 1

    if not counts:
        return None

    main_tid = max(counts, key=counts.get)
    print("[build_common_format] tid counts:", dict(counts), "-> main_tid:", main_tid)
    return main_tid


def filter_to_single_person(data: dict) -> dict:
    """
    Ensure each frame has at most one person.

    - If any tids exist:
        keep only the tid that appears in the most frames.
    - If all tids are null:
        * if only one person in frame -> keep it
        * if multiple:
            - if bbox exist and non-null -> keep one with largest bbox area
            - if no valid bbox -> keep the first person
    """
    if _has_any_tid(data):
        main_tid = _select_main_tid(data)
    else:
        main_tid = None

    for frame in data.get("frames", []):
        people = frame.get("people", [])
        if not people:
            continue

        if main_tid is not None:
            filtered = [p for p in people if p.get("tid") == main_tid]
            frame["people"] = filtered
        else:
            # No tids at all
            if len(people) == 1:
                # already only one
                frame["people"] = [people[0]]
            else:
                # multiple people, maybe some with bbox=None
                def area(p):
                    bbox = p.get("bbox")
                    if not bbox or len(bbox) != 4:
                        return 0.0
                    _, _, w, h = bbox
                    return float(w) * float(h)

                try:
                    best = max(people, key=area)
                except ValueError:
                    # somehow empty; just keep none
                    best = None

                frame["people"] = [best] if best is not None else []

    return data


# --------- joint + height utilities ---------

def _get_joint_from_kpts(
    kpts: list[float],
    joint_name: str,
    joint_names: list[str],
) -> tuple[float, float, float]:
    """
    Extract (x, y, conf) for a joint from a flat kpts list:
    [x0, y0, c0, x1, y1, c1, ...] using given joint_names ordering.
    """
    idx = joint_names.index(joint_name)
    x = float(kpts[3 * idx + 0])
    y = float(kpts[3 * idx + 1])
    c = float(kpts[3 * idx + 2])
    return x, y, c


def _compute_heights_px(
    joints_dict: dict,
    bbox: list[float] | None,
) -> tuple[float | None, float | None]:
    """
    Compute:
      - eye_to_ankle: vertical distance between average eye y and average ankle y
      - bbox_h: height of bounding box

    Returns (eye_to_ankle, bbox_h), each can be None if not available.
    """
    # Eye y: mean of confident left/right eye
    eyes: list[float] = []
    for name in ["left_eye", "right_eye"]:
        j = joints_dict.get(name)
        if j is not None and j["conf"] is not None and j["conf"] > 0.3:
            eyes.append(j["y"])
    eye_y = sum(eyes) / len(eyes) if eyes else None

    # Ankle y: mean of confident left/right ankle
    ankles: list[float] = []
    for name in ["left_ankle", "right_ankle"]:
        j = joints_dict.get(name)
        if j is not None and j["conf"] is not None and j["conf"] > 0.3:
            ankles.append(j["y"])
    ankle_y = sum(ankles) / len(ankles) if ankles else None

    if eye_y is not None and ankle_y is not None:
        eye_to_ankle = abs(ankle_y - eye_y)
    else:
        eye_to_ankle = None

    if bbox is not None and len(bbox) == 4:
        _, _, _, h = bbox
        bbox_h = float(h)
    else:
        bbox_h = None

    return eye_to_ankle, bbox_h


# --------- main conversion function ---------

def build_common_format(raw_data: dict, camera_id: str = "cam1") -> dict:
    """
    Convert one raw keypoints.json into the common format described at top.
    Supports both coco17 and mp33 layouts.
    """
    # figure out which layout we have
    layout = str(raw_data.get("layout", "coco17")).lower()
    if layout == "coco17":
        joint_names = COCO17_NAMES
    elif layout == "mp33":
        joint_names = MP33_NAMES
    else:
        print(f"[build_common_format] WARNING: unknown layout '{layout}', defaulting to coco17")
        joint_names = COCO17_NAMES

    # 1) keep only main person
    data = filter_to_single_person(raw_data)

    frames_out: list[dict] = []

    for frame in data.get("frames", []):
        frame_idx = frame["frame_idx"]
        ms = frame["ms"]
        people = frame.get("people", [])

        if not people:
            frames_out.append({
                "frame_idx": frame_idx,
                "ms": ms,
                "height_px": {
                    "eye_to_ankle": None,
                    "bbox_h": None,
                },
                "bbox": None,
                "joints": {},
            })
            continue

        person = people[0]
        kpts = person["kpts"]
        bbox = person.get("bbox", None)

        # extract our selected joints
        joints_dict: dict[str, dict] = {}
        for jname in TARGET_JOINTS:
            # skip if this joint name does not exist in this layout
            if jname not in joint_names:
                continue
            try:
                x, y, c = _get_joint_from_kpts(kpts, jname, joint_names)
            except (ValueError, IndexError):
                # joint name not present or kpts length mismatch
                continue
            joints_dict[jname] = {"x": x, "y": y, "conf": c}

        # compute height estimates
        eye_to_ankle, bbox_h = _compute_heights_px(joints_dict, bbox)

        frames_out.append({
            "frame_idx": frame_idx,
            "ms": ms,
            "height_px": {
                "eye_to_ankle": eye_to_ankle,
                "bbox_h": bbox_h,
            },
            "bbox": (
                {
                    "x": float(bbox[0]),
                    "y": float(bbox[1]),
                    "w": float(bbox[2]),
                    "h": float(bbox[3]),
                }
                if bbox is not None and len(bbox) == 4
                else None
            ),
            "joints": joints_dict,
        })

    out = {
        "schema_version": "pose_common_v1",
        "method": data.get("method", "unknown"),
        "camera": camera_id,
        "video": data.get("video"),
        "fps": data.get("fps"),
        "width": data.get("width"),
        "height": data.get("height"),
        "joints": TARGET_JOINTS,
        "person_id": "main",
        "frames": frames_out,
    }

    return out


# --------- file-level wrapper + CLI ---------

def convert_file(in_path: str | Path, out_path: str | Path, camera_id: str) -> None:
    raw = load_json(in_path)
    common = build_common_format(raw, camera_id=camera_id)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(common, f, indent=2)
    print(f"[build_common_format] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw keypoints.json (coco17/mp33) to common analysis format."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input keypoints.json"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output common-format JSON"
    )
    parser.add_argument(
        "--camera", "-c", required=True,
        help="Camera id label (e.g. cam1, cam2)"
    )

    args = parser.parse_args()
    convert_file(args.input, args.output, camera_id=args.camera)


if __name__ == "__main__":
    main()
