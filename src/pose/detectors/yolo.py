# src/pose/detectors/yolo.py
from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None
    _IMPORT_ERROR = e


class YOLODetector:
    """
    Person detector wrapper around Ultralytics YOLOv8.

    Returns detections as XYWH pixel boxes with scores:
        [[x, y, w, h, score], ...]

    Notes
    -----
    - Assumes COCO classes; filters to class 0 (person).
    - `frame` must be a BGR uint8 image (OpenCV style).
    - Model path defaults to 'models/yolov8n.pt'.
    """

    def __init__(
        self,
        weights: str = "models/yolov8n.pt",
        conf: float = 0.5,
        iou: float = 0.5,
        device: Optional[str] = None,   # "cuda", "cpu", or "mps"
        imgsz: int = 640,
        half: bool = False,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed or failed to import. "
                f"Original error: {_IMPORT_ERROR}"
            )

        self.model = YOLO(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device  # ultralytics handles None → auto
        self.half = bool(half)

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Run person detection on a single frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) uint8, BGR

        Returns
        -------
        List[[x, y, w, h, score]]  in pixels (float)
        """
        # Ultralytics handles resizing/letterboxing internally.
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )

        out: List[Tuple[float, float, float, float, float]] = []
        if not results:
            return out

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xywh is None:
            return out

        # xywh in pixels; cls indices; confidence per box
        xywh = boxes.xywh.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xywh), dtype=int)
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xywh), dtype=float)

        # COCO class 0 == "person"
        for (x, y, w, h), c, s in zip(xywh, cls, conf):
            if c == 0:
                out.append((float(x), float(y), float(w), float(h), float(s)))

        return out
