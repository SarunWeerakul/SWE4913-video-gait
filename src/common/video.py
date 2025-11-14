# src/common/video.py
"""
Lightweight OpenCV helpers shared across pipelines.

Functions:
- fps_of(path) -> float
- read_info(path) -> dict(width, height, fps, frames)
- iter_frames(path) -> yields (idx, frame, t_sec)
- make_writer(out_path, width, height, fps=30) -> cv2.VideoWriter
- write_overlay(in_path, out_path, draw_fn) -> None
"""

from __future__ import annotations
import os
from pathlib import Path
import cv2


# --- tiny internal util -------------------------------------------------------

def _ensure_dir(p: str | os.PathLike | None) -> None:
    """Create parent directory if needed (no-op for empty/None)."""
    if not p:
        return
    Path(p).mkdir(parents=True, exist_ok=True)


# --- public API ---------------------------------------------------------------

def fps_of(path: str | os.PathLike) -> float:
    """Return FPS of a video file (0.0 on failure)."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        return float(fps)
    finally:
        cap.release()


def read_info(path: str | os.PathLike) -> dict:
    """Read (width, height, fps, frames) for a video."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"width": w, "height": h, "fps": fps, "frames": n}
    finally:
        cap.release()


def iter_frames(path: str | os.PathLike):
    """
    Lazy generator over video frames.
    Yields: (idx, frame_bgr, t_sec)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_sec = (idx / fps) if fps > 0 else 0.0
            yield idx, frame, t_sec
            idx += 1
    finally:
        cap.release()


def make_writer(out_path: str | os.PathLike, width: int, height: int, fps: float = 30.0):
    """
    Create an MP4 writer (mp4v). Caller must .release() it.
    """
    _ensure_dir(os.path.dirname(str(out_path)) or ".")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))


def write_overlay(in_path: str | os.PathLike,
                  out_path: str | os.PathLike,
                  draw_fn):
    """
    Create an overlay video by applying draw_fn(frame, idx, t_sec) per frame.

    draw_fn: function that mutates/returns the frame to write.
             Signature: (frame_bgr, idx:int, t_sec:float) -> frame_bgr
    """
    info = read_info(in_path)
    writer = make_writer(out_path, info["width"], info["height"], info["fps"] or 30.0)
    try:
        for idx, frame, t in iter_frames(in_path):
            out = draw_fn(frame, idx, t)
            # allow draw_fn to return None (means "use original")
            if out is None:
                out = frame
            writer.write(out)
    finally:
        writer.release()
