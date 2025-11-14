# src/common/schema.py
from __future__ import annotations
from typing import Iterable, List, Dict, Any

SCHEMA_VERSION = "1.0"

# ---- Keypoint layouts --------------------------------------------------------
COCO17 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
COCO17_EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,1),(0,2),(1,3),(2,4),
]

# ---- Builders ----------------------------------------------------------------
def make(video: str, fps: float, w: int, h: int, frames: List[Dict[str, Any]],
         layout: str = "coco17", method: str | None = None, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Top-level record."""
    return {
        "schema_version": SCHEMA_VERSION,
        "layout": layout,              # e.g., "coco17" (so readers know how to interpret keypoints)
        "method": method,              # "yolo" | "topdown" | "posepipe" (optional)
        "video": video,
        "fps": float(fps),
        "width": int(w),
        "height": int(h),
        "frames": frames,
        "meta": meta or {},
    }

def frame(idx: int, ms: float, people: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-frame record; ms is timestamp in milliseconds from start."""
    return {"frame_idx": int(idx), "ms": float(ms), "people": people}

def person(kpts: Iterable[float], score: float | None = None, tid: int | None = None,
           bbox: Iterable[float] | None = None) -> Dict[str, Any]:
    """
    Person record:
      kpts: [x1,y1,conf1, x2,y2,conf2, ...] (len == 17*3 for COCO17)
      score: detector/pose score (optional)
      tid: tracker id if available (optional)
      bbox: [x, y, w, h] (optional)
    """
    kpts_list = list(map(float, kpts))
    return {"kpts": kpts_list, "score": (None if score is None else float(score)),
            "tid": tid, "bbox": (list(map(float, bbox)) if bbox is not None else None)}

# ---- Validation helpers ------------------------------------------------------
def validate_person(p: Dict[str, Any], expected_points: int = 17) -> None:
    k = p.get("kpts", [])
    if len(k) != expected_points * 3:
        raise ValueError(f"person.kpts must have {expected_points*3} values, got {len(k)}")

def validate_frame(f: Dict[str, Any], expected_points: int = 17) -> None:
    if "frame_idx" not in f or "ms" not in f or "people" not in f:
        raise ValueError("frame must contain keys: frame_idx, ms, people")
    for p in f["people"]:
        validate_person(p, expected_points)

def validate_top(doc: Dict[str, Any]) -> None:
    if doc.get("layout") == "coco17" and len(COCO17) != 17:
        raise ValueError("internal COCO17 mismatch")
    if "frames" not in doc:
        raise ValueError("top-level doc missing 'frames'")

# ---- Convenience -------------------------------------------------------------
def ms_from_frame_idx(idx: int, fps: float) -> float:
    return (1000.0 * idx / fps) if fps > 0 else 0.0
