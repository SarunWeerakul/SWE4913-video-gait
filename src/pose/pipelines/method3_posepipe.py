# src/pose/pipelines/method3_posepipe.py
from __future__ import annotations
from pathlib import Path
import os
from typing import List

import numpy as np
import cv2

from common.video import read_info, iter_frames, make_writer
from common.schema import (
    make as make_doc,
    frame as make_frame,
    person as make_person,
    ms_from_frame_idx,
)
from common.io import ensure_dir, write_json
from common.log import get_logger
from pose.utils import draw_skeleton


# ---- MediaPipe setup ---------------------------------------------------------

def _mp_init(model_path: str):
    """
    Initialize MediaPipe Tasks Pose Landmarker (video mode).
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except Exception as e:
        raise ImportError(
            "MediaPipe Tasks not available. Install with:\n"
            "  pip install mediapipe\n"
            f"Original error: {e}"
        ) from e

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download pose_landmarker_full.task from the official MediaPipe release and place it under models/."
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


def _lm33_to_flat_xyc(lms, width: int, height: int) -> List[float]:
    """
    Convert MediaPipe 33 landmarks (normalized) to [x,y,conf]*33 in pixels.
    Uses `visibility` as the confidence if present; defaults to 0.5 otherwise.
    """
    flat: List[float] = []
    for lm in lms:
        x = float(lm.x) * width
        y = float(lm.y) * height
        c = float(getattr(lm, "visibility", 0.5))
        flat.extend([x, y, c])
    return flat


# ---- Pipeline entry ----------------------------------------------------------

def main(in_mp4: str, out_json: str, vis_dir: str, n_overlay: int, out_mp4: str) -> None:
    """
    MediaPipe Pose pipeline (33 landmarks). Produces:
      - out_json: timeline with frames -> people[{kpts:[x,y,conf]*33}]
      - out_mp4 : overlay video with drawn skeleton
      - vis_dir : optional PNG frames (up to n_overlay)
    """
    log = get_logger("pose.method3_posepipe", "INFO")

    # Ensure outputs
    ensure_dir(os.path.dirname(out_json) or ".")
    ensure_dir(os.path.dirname(out_mp4) or ".")
    ensure_dir(vis_dir)

    # Video info
    info = read_info(in_mp4)
    fps = info["fps"] or 30.0
    W, H = info["width"], info["height"]

    # Init MediaPipe
    model_path = "models/pose_landmarker_full.task"
    landmarker, mp = _mp_init(model_path)

    # Writers / accumulators
    writer = make_writer(out_mp4, W, H, fps)
    frames_out = []

    try:
        for idx, frame_bgr, _t in iter_frames(in_mp4):
            # MP expects RGB (uint8, HxWx3) and contiguous
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)

            ts_ms = int(ms_from_frame_idx(idx, fps))

            # IMPORTANT: use positional args for older MP builds
            mp_frame = mp.Image(mp.ImageFormat.SRGB, frame_rgb)

            result = landmarker.detect_for_video(mp_frame, ts_ms)

            people = []
            if result.pose_landmarks:
                # num_poses=1 → first pose only
                kpts = _lm33_to_flat_xyc(result.pose_landmarks[0], W, H)
                # bbox unknown from MP; store whole frame as a placeholder bbox
                people = [{"kpts": kpts, "score": 1.0, "bbox_xywh": [W / 2.0, H / 2.0, W, H, 1.0]}]

            # JSON record
            ppl_records = [make_person(p["kpts"], score=p["score"]) for p in people]
            frames_out.append(make_frame(idx, ts_ms, ppl_records))

            # Overlay drawing (limit to n_overlay)
            if idx < n_overlay and people:
                draw_skeleton(frame_bgr, people[0]["kpts"], conf_th=0.3)  # auto-detects MP-33 layout
                # optional debug still
                try:
                    Path(vis_dir).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(Path(vis_dir) / f"frame_{idx:06d}.png"), frame_bgr)
                except Exception:
                    pass

            writer.write(frame_bgr)

    finally:
        writer.release()

    # Top-level JSON (note: 33-point layout)
    doc = make_doc(
        video=in_mp4,
        fps=fps,
        w=W,
        h=H,
        frames=frames_out,
        method="posepipe",
        layout="mp33",
    )
    write_json(out_json, doc)

    log.info(f"Wrote JSON: {out_json}")
    log.info(f"Wrote overlay MP4: {out_mp4}")
