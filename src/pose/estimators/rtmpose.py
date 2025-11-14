# src/pose/estimators/rtmpose.py
from __future__ import annotations

import os
import importlib
from typing import List, Tuple

import numpy as np
import cv2
from packaging.version import parse as V


class RTMPose:
    """
    RTMPose estimator using MMPose >= 1.0 inferencer (CPU/MPS friendly).
    Input: list of det boxes in full-frame pixels as (cx, cy, w, h, score).
    Output: list of dicts:
      {
        "kpts": [x,y,conf]*17,          # COCO-17, full-frame coords
        "score": float,                 # usually the det score
        "bbox_xywh": [cx,cy,w,h,score]
      }
    """

    def __init__(self, device: str | None = None, model_alias: str | None = None):
        # Ensure we're on the 1.x API
        import mmpose
        ver = mmpose.__version__
        if V(ver) < V("1.0.0"):
            raise RuntimeError(
                f"mmpose >= 1.0 required, found {ver}. "
                "Install: pip install -U 'mmpose>=1.0' 'mmengine>=0.10' 'mmcv-lite>=2.0'"
            )

        # Import the inferencer directly to avoid package-level imports
        # that pull transformer heads and compiled mmcv ops.
        try:
            infer_mod = importlib.import_module(
                "mmpose.apis.inferencers.mmpose_inferencer"
            )
            MMPoseInferencer = getattr(infer_mod, "MMPoseInferencer")
        except Exception as e:
            raise ImportError(
                "Failed to import MMPoseInferencer directly. "
                "Ensure: mmpose>=1.0, mmengine>=0.10, mmcv-lite>=2.0 are installed."
            ) from e

        alias = model_alias or os.getenv("RTM_ALIAS", "rtmpose-s")
        self.inferencer = MMPoseInferencer(pose2d=alias, device=device or "cpu")

    # ------------------------ helpers ------------------------

    @staticmethod
    def _xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        cx, cy, w, h = box
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return x1, y1, x2, y2

    @staticmethod
    def _clip(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1: x2 = min(W - 1, x1 + 1)
        if y2 <= y1: y2 = min(H - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _pack_coco17_xyc(kpts_xy: np.ndarray, kpts_score: np.ndarray) -> List[float]:
        out: List[float] = []
        for (x, y), c in zip(kpts_xy.tolist(), kpts_score.tolist()):
            out.extend([float(x), float(y), float(c)])
        return out

    # ------------------------ main API ------------------------

    def infer(
        self,
        frame_bgr: np.ndarray,
        dets_cxcywh: List[Tuple[float, float, float, float, float]],
    ) -> List[dict]:
        H, W = frame_bgr.shape[:2]
        people: List[dict] = []
        if not dets_cxcywh:
            return people

        for (cx, cy, w, h, det_conf) in dets_cxcywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy((cx, cy, w, h))
            x1, y1, x2, y2 = self._clip(x1, y1, x2, y2, W, H)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Run the inferencer on the crop
            gen = self.inferencer(inputs=rgb, return_datasamples=True, show=False)
            result = next(gen)  # one image -> one yield

            kpts_xy = None
            kpts_sc = None

            # Preferred path: DataSample objects
            try:
                data_samples = result.get("data_samples", None)
                if data_samples:
                    inst = data_samples[0].pred_instances
                    if getattr(inst, "keypoints", None) is not None and len(inst.keypoints) > 0:
                        kpts_xy = inst.keypoints[0]  # (K,2)
                        kpts_sc = getattr(inst, "keypoint_scores", None)
                        if kpts_sc is not None:
                            kpts_sc = kpts_sc[0]      # (K,)
            except Exception:
                pass

            # Fallback: predictions dict
            if kpts_xy is None:
                preds = result.get("predictions", None)
                if preds:
                    p0 = preds[0]
                    if "keypoints" in p0 and len(p0["keypoints"]) > 0:
                        kpts_xy = np.asarray(p0["keypoints"][0], dtype=np.float32)  # (K,2)
                        if "keypoint_scores" in p0:
                            kpts_sc = np.asarray(p0["keypoint_scores"][0], dtype=np.float32)

            if kpts_xy is None:
                # no pose found in this crop
                continue

            K = kpts_xy.shape[0]
            if kpts_sc is None:
                kpts_sc = np.ones((K,), dtype=np.float32) * 0.5

            # Map crop -> full-frame
            kpts_xy_full = kpts_xy.copy()
            kpts_xy_full[:, 0] += x1
            kpts_xy_full[:, 1] += y1

            packed = self._pack_coco17_xyc(kpts_xy_full, kpts_sc)
            people.append({
                "kpts": packed,
                "score": float(det_conf),
                "bbox_xywh": [float(cx), float(cy), float(w), float(h), float(det_conf)],
            })

        return people
