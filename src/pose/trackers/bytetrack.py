# src/pose/trackers/bytetrack.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """Center (x,y,w,h) → corners (x1,y1,x2,y2)."""
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return x1, y1, x2, y2


class _CentroidFallback:
    """
    Minimal tracker when boxmot is unavailable.
    Matches detections to previous track centers by nearest neighbor.
    """
    def __init__(self, max_dist: float = 80.0):
        self.max_dist = max_dist
        self.next_id = 1
        self.tracks: Dict[int, Tuple[float, float]] = {}  # id -> (cx, cy)

    def update(self, dets_xywh: np.ndarray) -> List[Dict]:
        out = []
        if dets_xywh.size == 0:
            return out

        # compute detection centers
        det_centers = dets_xywh[:, :2]  # (x,y) are already centers
        det_used = np.zeros(len(det_centers), dtype=bool)
        new_tracks: Dict[int, Tuple[float, float]] = {}

        # greedy match existing tracks to nearest detection
        for tid, (tx, ty) in self.tracks.items():
            if len(det_centers) == 0:
                continue
            dists = np.linalg.norm(det_centers - np.array([tx, ty])[None, :], axis=1)
            j = int(np.argmin(dists))
            if det_used[j] or dists[j] > self.max_dist:
                continue
            det_used[j] = True
            cx, cy = det_centers[j]
            w, h = dets_xywh[j, 2], dets_xywh[j, 3]
            new_tracks[tid] = (float(cx), float(cy))
            out.append({"id": tid, "bbox_xywh": dets_xywh[j].tolist(), "score": float(dets_xywh[j, 4])})

        # spawn new tracks for unmatched detections
        for j in range(len(det_centers)):
            if det_used[j]:
                continue
            tid = self.next_id
            self.next_id += 1
            cx, cy = det_centers[j]
            new_tracks[tid] = (float(cx), float(cy))
            out.append({"id": tid, "bbox_xywh": dets_xywh[j].tolist(), "score": float(dets_xywh[j, 4])})

        self.tracks = new_tracks
        return out


class Tracker:
    """
    Thin wrapper over boxmot's ByteTrack with a graceful fallback.

    Input detections per frame: list of (x, y, w, h, score) in **XYWH center** pixels.
    Output tracks: list of dicts:
        {"id": int, "bbox_xywh": [x,y,w,h,score], "score": float}
    """
    def __init__(
        self,
        use_fallback_if_missing: bool = True,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        conf_thresh: float = 0.5,
        max_age: int = 30,
        device: Optional[str] = None,
    ):
        self._use_fallback = use_fallback_if_missing
        self._bt = None

        try:
            from boxmot.tracker.byte_tracker.byte_tracker import BYTETracker
            from boxmot.utils import BOXMOT
            self._bt = BYTETracker(
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                conf_thres=conf_thresh,
                max_age=max_age,
                device=device or "cpu",
            )
            self._boxmot_ok = True
        except Exception:
            self._boxmot_ok = False
            if not self._use_fallback:
                raise
            self._fb = _CentroidFallback()

    def update(self, detections_xywh: List[Tuple[float, float, float, float, float]], frame_id: int = 0) -> List[Dict]:
        """
        Parameters
        ----------
        detections_xywh : list of (x, y, w, h, score) center format (pixels)
        frame_id : int (optional, ignored by fallback)

        Returns
        -------
        List[dict] with keys: id, bbox_xywh, score
        """
        if len(detections_xywh) == 0:
            return []

        dets = np.asarray(detections_xywh, dtype=float)
        if dets.ndim != 2 or dets.shape[1] < 5:
            raise ValueError("detections_xywh must be Nx5 (x,y,w,h,score).")

        if self._boxmot_ok and self._bt is not None:
            # boxmot expects xyxy + score + class (last col)
            xyxy = np.array([_xywh_to_xyxy(*row[:4]) for row in dets], dtype=float)
            scores = dets[:, 4:5]
            cls = np.zeros((len(dets), 1), dtype=float)  # class 0 = person
            det_for_bt = np.concatenate([xyxy, scores, cls], axis=1)  # Nx6

            tracks = self._bt.update(det_for_bt, frame_id)  # returns list of Track objects or ndarray depending on version
            out: List[Dict] = []
            # Handle both old/new boxmot return shapes
            for tr in tracks:
                if hasattr(tr, "tlbr"):  # object-like (tlbr + track_id)
                    x1, y1, x2, y2 = tr.tlbr
                    tid = int(tr.track_id)
                    score = float(getattr(tr, "score", 1.0))
                else:  # ndarray [x1,y1,x2,y2,track_id,score]
                    x1, y1, x2, y2, tid, score = map(float, tr[:6])
                    tid = int(tid)
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                out.append({"id": tid, "bbox_xywh": [cx, cy, w, h, score], "score": score})
            return out

        # Fallback path
        return self._fb.update(dets)
