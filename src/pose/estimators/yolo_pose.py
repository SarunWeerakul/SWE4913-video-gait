# src/pose/estimators/yolo_pose.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None
    _IMPORT_ERROR = e


class YOLOPose:
    """
    Wrapper for Ultralytics YOLOv8-Pose.
    Returns a list of people dicts per frame:
        {
          "kpts": [x1,y1,c1, x2,y2,c2, ...]  # len == 17*3
          "score": float,                    # detector/pose score
          "bbox_xywh": [cx,cy,w,h,score],    # center-format pixels
        }
    """

    def __init__(
        self,
        weights: str = "models/yolov8n-pose.pt",
        conf: float = 0.5,
        iou: float = 0.5,
        device: Optional[str] = None,   # "cuda", "cpu", or "mps"
        imgsz: int = 640,
        half: bool = False,
        max_persons: Optional[int] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed or failed to import. "
                f"Original error: {_IMPORT_ERROR}"
            )
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.imgsz = int(imgsz)
        self.half = bool(half)
        self.max_persons = max_persons

    def infer(self, frame) -> List[Dict]:
        """
        Run pose on a single BGR uint8 frame.
        Returns: list of {"kpts": [...], "score": float, "bbox_xywh": [cx,cy,w,h,score]}.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        people: List[Dict] = []

        # Boxes (xywh) and confidences
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xywh is None:
            return people

        xywh = boxes.xywh  # tensor Nx4
        scores = boxes.conf if getattr(boxes, "conf", None) is not None else None

        # Keypoints (varies slightly by Ultralytics version)
        kobj = getattr(r, "keypoints", None)
        has_kpts = kobj is not None
        k_xy = None
        k_conf = None
        if has_kpts:
            # Prefer pixel coords
            if hasattr(kobj, "xy") and kobj.xy is not None:
                k_xy = kobj.xy  # list[ndarray] or tensor (N,17,2)
            elif hasattr(kobj, "xyn") and kobj.xyn is not None:
                # normalized -> fallback to normalized; caller can scale if needed
                k_xy = kobj.xyn
            # confidences (optional)
            if hasattr(kobj, "conf") and kobj.conf is not None:
                k_conf = kobj.conf

        N = len(xywh)
        order = range(N)
        if self.max_persons:
            # sort by score desc and keep top-k
            scr_np = scores.cpu().numpy() if scores is not None else np.ones(N, dtype=float)
            order = np.argsort(-scr_np)[: self.max_persons]

        for i in order:
            # bbox center xywh + score
            cx, cy, w, h = [float(v) for v in xywh[i].cpu().numpy().tolist()]
            score = float(scores[i].item()) if scores is not None else 1.0
            bbox_xywh = [cx, cy, w, h, score]

            # keypoints -> flat [x,y,conf]*17
            if has_kpts:
                # shape handling: tensor or ndarray, both to numpy
                if isinstance(k_xy, list):
                    xy = np.asarray(k_xy[i])
                else:
                    xy = k_xy[i].cpu().numpy()
                # confidences per keypoint if available
                if k_conf is not None:
                    if isinstance(k_conf, list):
                        kc = np.asarray(k_conf[i])
                    else:
                        kc = k_conf[i].cpu().numpy()
                    if kc.ndim == 1:
                        kc = kc[:, None]
                else:
                    # fallback: use detection score for all joints
                    kc = np.full((xy.shape[0], 1), score, dtype=float)

                if xy.ndim == 2 and xy.shape[1] == 2:
                    kpts_flat = np.concatenate([xy, kc], axis=1).reshape(-1).tolist()
                elif xy.ndim == 2 and xy.shape[1] == 3:
                    # already (x,y,conf)
                    kpts_flat = xy.reshape(-1).tolist()
                else:
                    # unexpected; skip this person
                    continue
            else:
                # no keypoints returned (shouldn't happen for *pose*); skip
                continue

            people.append({"kpts": kpts_flat, "score": score, "bbox_xywh": bbox_xywh})

        return people
