# src/pose/pipelines/method1_yolo.py
from __future__ import annotations

import os
from pathlib import Path

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
from pose.estimators.yolo_pose import YOLOPose  # expects models/yolov8n-pose.pt


def main(in_mp4: str, out_json: str, vis_dir: str, n_overlay: int, out_mp4: str) -> None:
    """
    Single-stage YOLOv8-Pose pipeline.

    Inputs
    ------
    in_mp4    : input video
    out_json  : path to write keypoint timeline (JSON)
    vis_dir   : folder to save up to n_overlay debug frames
    n_overlay : number of frames to draw/save for debugging
    out_mp4   : path to write overlay video (MP4)
    """
    log = get_logger("pose.method1_yolo", "INFO")

    # Ensure output dirs exist
    ensure_dir(os.path.dirname(out_json) or ".")
    ensure_dir(os.path.dirname(out_mp4) or ".")
    ensure_dir(vis_dir)

    # Video info
    info = read_info(in_mp4)
    fps = info["fps"] or 30.0
    width, height = info["width"], info["height"]

    # Model
    est = YOLOPose(weights="models/yolov8n-pose.pt", conf=0.5, iou=0.5)

    # Writers / accumulators
    writer = make_writer(out_mp4, width, height, fps)
    frames_out = []

    try:
        for idx, frame_bgr, t_sec in iter_frames(in_mp4):
            # Inference: list of {"kpts":[...], "score": float, "bbox_xywh":[cx,cy,w,h,score]}
            people = est.infer(frame_bgr)

            # Build schema records
            ppl_records = [
                make_person(p["kpts"], score=p["score"], bbox=p["bbox_xywh"][:4])
                for p in people
            ]
            frames_out.append(
                make_frame(idx, ms_from_frame_idx(idx, fps), ppl_records)
            )

            # Draw overlay (and optionally save debug frames)
            if idx < n_overlay:
                for p in people:
                    draw_skeleton(frame_bgr, p["kpts"], conf_th=0.3)
                # save PNG debug frame
                try:
                    from cv2 import imwrite  # lazy import cv2 only for saving stills
                    debug_dir = Path(vis_dir)
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    imwrite(str(debug_dir / f"frame_{idx:06d}.png"), frame_bgr)
                except Exception:
                    pass  # best-effort; overlay video still gets the drawing

            writer.write(frame_bgr)

    finally:
        writer.release()

    # Top-level JSON
    doc = make_doc(
        video=in_mp4,
        fps=fps,
        w=width,
        h=height,
        frames=frames_out,
        method="yolo",
        layout="coco17",
    )
    write_json(out_json, doc)

    log.info(f"Wrote JSON: {out_json}")
    log.info(f"Wrote overlay MP4: {out_mp4}")
