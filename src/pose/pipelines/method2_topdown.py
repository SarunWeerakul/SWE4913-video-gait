# src/pose/pipelines/method2_topdown.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

from common.video import read_info, iter_frames, make_writer
from common.schema import (
    make as make_doc,
    frame as make_frame,
    person as make_person,
    ms_from_frame_idx,
)
from common.io import write_json, ensure_dir
from common.log import get_logger
from pose.utils import draw_skeleton
from pose.detectors.yolo import YOLODetector


# ---------------- MediaPipe pose (mp33) ----------------

def _mp_init(model_path: str):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError(
            "MediaPipe Tasks not available. Install:\n"
            "  pip install mediapipe\n"
            f"Original error: {e}"
        ) from e

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Download pose_landmarker_full.task into models/ ."
        )

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
    return landmarker, mp


def _xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    cx, cy, w, h = box
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return x1, y1, x2, y2


def _clip(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def _lm33_to_flat_xyc(lms, W: int, H: int, offx: int, offy: int) -> List[float]:
    """Map MediaPipe 33 normalized landmarks from crop to full-frame [x,y,c]*33."""
    out: List[float] = []
    for lm in lms:
        x = offx + float(lm.x) * W
        y = offy + float(lm.y) * H
        c = float(getattr(lm, "visibility", 0.5))
        out.extend([x, y, c])
    return out


# ---------------- pipeline ----------------

def main(in_mp4: str, out_json: str, vis_dir: str, n_overlay: int, out_mp4: str) -> None:
    """
    Top-down pipeline (YOLO + MediaPipe Pose):
      1) Detect person bboxes (YOLO)
      2) Crop the best person
      3) Run MediaPipe Pose on the crop
      4) Map 33 keypoints back to full frame, write JSON + overlay
    """
    log = get_logger("pose.method2_topdown.mp33", "INFO")

    # outputs
    ensure_dir(os.path.dirname(out_json) or ".")
    ensure_dir(os.path.dirname(out_mp4) or ".")
    ensure_dir(vis_dir)

    # video info
    info = read_info(in_mp4)
    fps = info["fps"] or 30.0
    W, H = info["width"], info["height"]

    # detector (env overrides optional)
    det_conf = float(os.getenv("DET_CONF", "0.5"))
    det_iou  = float(os.getenv("DET_IOU", "0.5"))
    det_img  = int(os.getenv("DET_IMGSZ", "640"))
    det_wts  = os.getenv("DET_WEIGHTS", "models/yolov8n.pt")
    if det_wts and os.path.splitext(det_wts)[1] and not os.path.exists(det_wts):
        raise FileNotFoundError(
            f"YOLO weights not found at '{det_wts}'. "
            "Set DET_WEIGHTS or place yolov8n.pt under models/."
        )
    detector = YOLODetector(weights=det_wts, conf=det_conf, iou=det_iou, imgsz=det_img)

    # mediapipe pose
    mp_model = os.getenv("MP_POSE_TASK", "models/pose_landmarker_full.task")
    landmarker, mp = _mp_init(mp_model)

    writer = make_writer(out_mp4, W, H, fps)
    frames_out = []

    try:
        for idx, frame_bgr, _t in iter_frames(in_mp4):
            ts_ms = int(ms_from_frame_idx(idx, fps))

            # detect
            dets = detector.detect(frame_bgr)  # [(cx,cy,w,h,score), ...]
            people = []

            # choose the best person bbox (you can extend to multi-person)
            best = None
            for (cx, cy, w, h, s) in dets:
                if best is None or s > best[-1]:
                    best = (cx, cy, w, h, s)

            if best is not None:
                cx, cy, w, h, s = best
                x1, y1, x2, y2 = _xywh_to_xyxy((cx, cy, w, h))
                x1, y1, x2, y2 = _clip(x1, y1, x2, y2, W, H)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    mp_img = mp.Image(mp.ImageFormat.SRGB, rgb)
                    res = landmarker.detect_for_video(mp_img, ts_ms)
                    if res.pose_landmarks:
                        kpts = _lm33_to_flat_xyc(res.pose_landmarks[0], W=(x2 - x1), H=(y2 - y1), offx=x1, offy=y1)
                        people = [{"kpts": kpts, "score": float(s), "bbox_xywh": [cx, cy, w, h, float(s)]}]

            # JSON
            ppl_records = [make_person(p["kpts"], score=p.get("score", 1.0)) for p in people]
            frames_out.append(make_frame(idx, ts_ms, ppl_records))

            # overlay (mp33)
            if idx < n_overlay and people:
                draw_skeleton(frame_bgr, people[0]["kpts"], conf_th=0.3)
                try:
                    Path(vis_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(vis_dir) / f"frame_{idx:06d}.png"), frame_bgr)
                except Exception:
                    pass

            writer.write(frame_bgr)

    finally:
        writer.release()

    # write JSON
    doc = make_doc(
        video=in_mp4,
        fps=fps,
        w=W,
        h=H,
        frames=frames_out,
        method="topdown_mp33",
        layout="mp33",
    )
    write_json(out_json, doc)

    log.info(f"Wrote JSON: {out_json}")
    log.info(f"Wrote overlay MP4: {out_mp4}")
