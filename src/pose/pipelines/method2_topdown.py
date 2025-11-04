#!/usr/bin/env python3
"""
Method 2: Top-down (YOLO person detector -> crop -> MediaPipe Pose)
CLI: <in.mp4> <out.json> <vis_dir> <n_overlay> <out.mp4>

- Detect person bbox per frame (YOLO)
- Crop with margin, run MediaPipe Pose on the crop
- Map 33 MP joints -> your COCO17 order (x,y,conf), scaled back to full-frame coords
- Keep the most-moving track (via ankle mid) across the sequence
"""
import os, sys, json, math, cv2, numpy as np
from collections import defaultdict

# --- deps ---
try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "Ultralytics not available. Install with:\n"
        "  pip install ultralytics\n"
        f"Import error: {e}"
    )

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ------------ Config ------------
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")  # change to local model if needed
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IMG = int(os.environ.get("YOLO_IMG", "640"))
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", "0.20"))   # 20% padding around bbox

COCO17 = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# Map COCO17 joints -> MediaPipe Pose (33) indices
MP = {
  "nose":0, "left_eye":2, "right_eye":5, "left_ear":7, "right_ear":8,
  "left_shoulder":11, "right_shoulder":12, "left_elbow":13, "right_elbow":14,
  "left_wrist":15, "right_wrist":16, "left_hip":23, "right_hip":24,
  "left_knee":25, "right_knee":26, "left_ankle":27, "right_ankle":28
}
COCO2MP = [MP[n] for n in COCO17]
L_ANK, R_ANK = 15, 16  # indices within COCO17 array

class EMA:
    def __init__(self, alpha=0.25): self.a=alpha; self.prev=None
    def step(self, v):
        if v is None: return self.prev
        self.prev = v if self.prev is None else (self.a*v + (1-self.a)*self.prev)
        return self.prev

def draw_skeleton(img, kpts, th=0.35):
    lines = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,11),(6,12),(5,6)]
    for x,y,c in kpts:
        if c is not None and c>=th and x is not None and y is not None:
            cv2.circle(img,(int(x),int(y)),3,(255,255,255),-1)
    for a,b in lines:
        xa,ya,ca=kpts[a]; xb,yb,cb=kpts[b]
        if None in (xa,ya,xb,yb): continue
        if (ca or 0)<th or (cb or 0)<th: continue
        cv2.line(img,(int(xa),int(ya)),(int(xb),int(yb)),(255,255,255),2)

def ankle_mid(kpts, th=0.35):
    la=kpts[L_ANK]; ra=kpts[R_ANK]
    if any(v is None for v in (la[0],la[1],ra[0],ra[1])): return None
    if (la[2] or 0)<th or (ra[2] or 0)<th: return None
    return ((la[0]+ra[0])/2.0,(la[1]+ra[1])/2.0)

def pick_person_bbox(yolo_result):
    """Return best person bbox (x1,y1,x2,y2,conf) or None."""
    best = None
    for b in getattr(yolo_result, "boxes", []):
        # Ultralytics: class id in b.cls, confid in b.conf
        try:
            cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            if cls != 0:  # class 0 = person in COCO
                continue
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
        except Exception:
            # older/newer structures fallback
            xyxy = np.asarray(b.xyxy).reshape(-1)
            x1,y1,x2,y2 = map(float, xyxy[:4])
            conf = float(getattr(b, "conf", 0.0))
            cls = int(getattr(b, "cls", 0))
            if cls != 0: continue
        if conf < YOLO_CONF: 
            continue
        if best is None or conf > best[-1]:
            best = (x1,y1,x2,y2,conf)
    return best

def expand_and_clip(x1,y1,x2,y2,W,H,margin=CROP_MARGIN):
    w = (x2 - x1); h = (y2 - y1)
    cx = x1 + w/2.0; cy = y1 + h/2.0
    s = max(w,h) * (1.0 + margin*2.0)  # square crop with margin
    nx1 = max(0.0, cx - s/2.0); ny1 = max(0.0, cy - s/2.0)
    nx2 = min(float(W), cx + s/2.0); ny2 = min(float(H), cy + s/2.0)
    return int(nx1), int(ny1), int(nx2), int(ny2)

def main():
    if len(sys.argv)!=6:
        print("Usage: method2_topdown.py <in.mp4> <out.json> <vis_dir> <n_overlay> <out.mp4>")
        sys.exit(1)
    in_mp4, out_json, vis_dir, n_overlay_str, out_mp4 = sys.argv[1:]
    n_overlay = int(n_overlay_str)

    # --- load models ---
    try:
        yolo = YOLO(YOLO_WEIGHTS)
    except Exception as e:
        raise SystemExit(
            f"Could not load YOLO weights '{YOLO_WEIGHTS}'.\n"
            "Install weights or set YOLO_WEIGHTS env var to a local .pt file.\n"
            f"Original error: {e}"
        )

    model_path = "models/pose_landmarker_full.task"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Download a MediaPipe Pose Landmarker .task model into models/."
        )
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {in_mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(vis_dir, exist_ok=True)
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), max(fps,1.0), (W,H))

    frames=[]
    fidx=0; saved=0

    # smoothing + path calc for the single selected track
    ema = [EMA(0.25) for _ in range(17)]
    path_len = 0.0
    prev_cent = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        ts_ms = int((fidx/(fps if fps>0 else 1))*1000)

        # --- detect person bbox (YOLO) on the full frame (BGR expected) ---
        det = yolo.predict(frame_bgr, imgsz=YOLO_IMG, conf=YOLO_CONF, verbose=False)[0]
        pick = pick_person_bbox(det)

        people = []
        canvas = np.zeros_like(frame_bgr)

        if pick is not None:
            x1,y1,x2,y2,_ = pick
            x1,y1,x2,y2 = expand_and_clip(x1,y1,x2,y2, W,H, CROP_MARGIN)

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                if result and result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    kpts = []
                    for j, mp_idx in enumerate(COCO2MP):
                        if mp_idx >= len(lms):
                            kpts.append([None,None,None]); continue
                        lm = lms[mp_idx]
                        # scale back to full-frame coordinates
                        x = None if lm.x is None else (x1 + lm.x * (x2 - x1))
                        y = None if lm.y is None else (y1 + lm.y * (y2 - y1))
                        c = getattr(lm, "visibility", None)
                        if x is not None and y is not None:
                            x = ema[j].step(x); y = ema[j].step(y)
                        kpts.append([x,y,c])

                    people.append({"kpts": kpts})
                    draw_skeleton(canvas, kpts)

                    # path update via ankle midpoint
                    cxy = ankle_mid(kpts)
                    if cxy is not None:
                        if prev_cent is not None:
                            dx = cxy[0]-prev_cent[0]; dy = cxy[1]-prev_cent[1]
                            path_len += math.hypot(dx, dy)
                        prev_cent = cxy

        vw.write(canvas)
        if saved < n_overlay:
            cv2.imwrite(os.path.join(vis_dir, f"{fidx:06d}.jpg"), canvas)
            saved += 1

        frames.append({"frame_idx": fidx, "ms": ts_ms, "people": people})
        fidx += 1

    cap.release(); vw.release()

    out = {"video": in_mp4, "fps": fps, "width": W, "height": H, "frames": frames}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f)

    print(f"Saved keypoints JSON -> {out_json}")
    if n_overlay>0:
        print(f"Saved overlays -> {vis_dir} (first {n_overlay} frames)")
    print(f"Saved skeleton-only MP4 -> {out_mp4}")

if __name__ == "__main__":
    main()
