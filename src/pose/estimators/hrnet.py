# src/pose/estimators/hrnet.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np


def _xywh_center_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return float(x1), float(y1), float(x2), float(y2)


class HRNetEstimator:
    """
    HRNet (top-down) pose estimator wrapper via MMPose.

    Preferred path (MMPose ≥1.0):
        HRNetEstimator(model="hrnet-w32", device="cpu")   # or "cuda:0"
    Legacy path (MMPose <1.0):
        HRNetEstimator(cfg="configs/topdown/...", checkpoint="hrnet_w32_coco_xxx.pth")

    infer(frame_bgr, det_xywh=[(cx,cy,w,h,score), ...]) -> List[{
        "kpts": [x,y,conf]*17,
        "score": float,
        "bbox_xywh": [cx,cy,w,h,score],
    }]
    """

    def __init__(
        self,
        model: str = "hrnet-w32",   # model name for MMPoseInferencer (≥1.0)
        device: Optional[str] = None,
        # legacy API fallback (only used if new API not available)
        cfg: Optional[str] = None,
        checkpoint: Optional[str] = None,
        bbox_thr: float = 0.0,
    ) -> None:
        self.device = device
        self.model_name = model
        self.legacy_cfg = cfg
        self.legacy_ckpt = checkpoint
        self.bbox_thr = float(bbox_thr)

        self._mode = None            # "inferencer" | "legacy"
        self._inferencer = None
        self._legacy_model = None

        # Try new API
        try:
            from mmpose.apis import MMPoseInferencer  # type: ignore
            self._inferencer = MMPoseInferencer(self.model_name, device=self.device)
            self._mode = "inferencer"
            return
        except Exception:
            pass

        # Try legacy API
        try:
            from mmpose.apis import init_model  # type: ignore
            if not (self.legacy_cfg and self.legacy_ckpt):
                raise ImportError(
                    "MMPose <1.0 detected: provide cfg and checkpoint for HRNet "
                    "(e.g., cfg='configs/topdown/hrnet/coco/hrnet_w32_coco_256x192.py', "
                    "checkpoint='hrnet_w32_coco_256x192-xxx.pth')."
                )
            self._legacy_model = init_model(self.legacy_cfg, self.legacy_ckpt, device=self.device or "cpu")
            self._mode = "legacy"
            return
        except Exception as e:
            raise ImportError(
                "Could not initialize HRNet via MMPose. Install mmpose + deps.\n"
                "New API (≥1.0): pip install 'mmpose>=1.0.0' mmengine 'mmcv>=2.0.0'\n"
                "Legacy  (<1.0): pip install 'mmpose<1.0' 'mmcv-full<2.0'\n"
                f"Original error: {e}"
            ) from e

    def infer(
        self,
        frame_bgr,
        det_xywh: List[Tuple[float, float, float, float, float]],
    ) -> List[Dict]:
        if len(det_xywh) == 0:
            return []
        if self._mode == "inferencer":
            return self._infer_inferencer(frame_bgr, det_xywh)
        elif self._mode == "legacy":
            return self._infer_legacy(frame_bgr, det_xywh)
        else:
            raise RuntimeError("HRNetEstimator not initialized.")

    # --- MMPose ≥1.0 path
    def _infer_inferencer(self, frame_bgr, det_xywh):
        from mmpose.apis import MMPoseInferencer  # type: ignore

        bboxes_xyxy = [list(_xywh_center_to_xyxy(*d[:4])) for d in det_xywh]
        inputs = dict(img=frame_bgr, bboxes=bboxes_xyxy)

        gen = self._inferencer(inputs, return_vis=False)
        out = next(gen)  # single frame result

        preds = out.get("predictions", [])
        people: List[Dict] = []

        # Some versions wrap in a list
        if isinstance(preds, list) and len(preds) == 1 and isinstance(preds[0], dict) and "keypoints" in preds[0]:
            preds = preds[0]

        keypoints = preds["keypoints"] if isinstance(preds, dict) else []
        # optional per-joint scores:
        keypoint_scores = preds.get("keypoint_scores", None) if isinstance(preds, dict) else None

        for i, (cx, cy, w, h, det_score) in enumerate(det_xywh):
            if i >= len(keypoints):
                continue
            k = np.asarray(keypoints[i])
            if k.ndim != 2:
                continue

            if k.shape[1] == 2:
                conf = np.full((k.shape[0], 1), float(det_score), dtype=float)
                k3 = np.concatenate([k, conf], axis=1)
            elif k.shape[1] == 3:
                k3 = k
            else:
                continue

            people.append({
                "kpts": k3.reshape(-1).astype(float).tolist(),
                "score": float(det_score),
                "bbox_xywh": [float(cx), float(cy), float(w), float(h), float(det_score)],
            })
        return people

    # --- MMPose <1.0 path
    def _infer_legacy(self, frame_bgr, det_xywh):
        from mmpose.apis import inference_topdown  # type: ignore

        person_results = []
        for (cx, cy, w, h, s) in det_xywh:
            x1, y1, x2, y2 = _xywh_center_to_xyxy(cx, cy, w, h)
            person_results.append({"bbox": np.array([x1, y1, x2, y2], dtype=float)})

        preds, _ = inference_topdown(self._legacy_model, frame_bgr, person_results)
        people: List[Dict] = []

        for (cx, cy, w, h, det_score), res in zip(det_xywh, preds):
            k = np.asarray(res["keypoints"], dtype=float)  # (K,3) [x,y,score]
            if k.ndim != 2 or k.shape[1] < 2:
                continue

            if k.shape[1] == 2:
                conf = np.full((k.shape[0], 1), float(det_score), dtype=float)
                k3 = np.concatenate([k, conf], axis=1)
            else:
                k3 = k[:, :3]

            people.append({
                "kpts": k3.reshape(-1).tolist(),
                "score": float(det_score),
                "bbox_xywh": [float(cx), float(cy), float(w), float(h), float(det_score)],
            })
        return people
