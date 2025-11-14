# src/pose/utils.py
from __future__ import annotations
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --- default skeleton graphs --------------------------------------------------

# COCO-17 keypoint indices (YOLOv8-pose):
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder,
# 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
# 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
COCO17_EDGES: List[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms + shoulders
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4),           # face linework
]

# MediaPipe Pose 33 landmarks — we draw a compact subset for clarity.
# Index map (abbrev): 11/12 shoulders, 13/14 elbows, 15/16 wrists,
# 23/24 hips, 25/26 knees, 27/28 ankles.
MP33_EDGES: List[Tuple[int, int]] = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # arms
    (23, 25), (25, 27), (24, 26), (26, 28),          # legs
]


def _as_list3(kpts: Sequence[float] | np.ndarray) -> List[Tuple[float, float, float]]:
    """Accept [x1,y1,c1, x2,y2,c2, ...] or Nx3 -> [(x,y,c), ...]."""
    if isinstance(kpts, np.ndarray):
        if kpts.ndim == 2 and kpts.shape[1] == 3:
            return [tuple(map(float, row)) for row in kpts.tolist()]
        kpts = kpts.flatten().tolist()
    # flat list
    assert len(kpts) % 3 == 0, "keypoints must be in triples (x,y,conf)"
    it = iter(kpts)
    return [(float(x), float(y), float(c)) for x, y, c in zip(it, it, it)]


def _auto_edges(n_points: int) -> List[Tuple[int, int]]:
    """Pick a sensible skeleton based on number of points."""
    if n_points == 17:
        return COCO17_EDGES
    if n_points == 33:
        return MP33_EDGES
    return []  # unknown layout; still draw keypoint dots


def draw_skeleton(
    img,
    kpts: Sequence[float] | np.ndarray,
    conf_th: float = 0.3,
    edges: Optional[List[Tuple[int, int]]] = None,
    draw_scores: bool = False,
):
    """
    Draw pose keypoints + skeleton lines onto `img` (BGR).

    Parameters
    ----------
    img : np.ndarray(HxWx3, uint8)
    kpts : [x1,y1,c1, x2,y2,c2, ...] or np.ndarray(N,3)
    conf_th : float
        Minimum confidence to draw a point/edge.
    edges : list[(i,j)] or None
        If None, auto-select COCO-17 or MP-33 based on #keypoints.
    draw_scores : bool
        If True, annotate tiny confidence values near points.

    Returns
    -------
    img : np.ndarray
        The same array reference with drawings applied.
    """
    h, w = img.shape[:2]
    pts = _as_list3(kpts)
    n = len(pts)
    G = edges if edges is not None else _auto_edges(n)

    # scale-friendly thickness/radius
    t = max(1, int(round(0.0035 * max(h, w))))   # line thickness
    r = max(2, int(round(0.0045 * max(h, w))))   # keypoint radius
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.3, 0.6 * (t / 2))                 # font scale

    # draw bones first
    for i, j in G:
        if i >= n or j >= n:
            continue
        x1, y1, c1 = pts[i]
        x2, y2, c2 = pts[j]
        if c1 >= conf_th and c2 >= conf_th:
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.line(img, p1, p2, (0, 255, 0), t, lineType=cv2.LINE_AA)

    # draw joints
    for idx, (x, y, c) in enumerate(pts):
        if c < conf_th:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, r, (0, 215, 255), -1, lineType=cv2.LINE_AA)
        if draw_scores:
            cv2.putText(img, f"{c:.2f}", (p[0] + 3, p[1] - 3), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return img
