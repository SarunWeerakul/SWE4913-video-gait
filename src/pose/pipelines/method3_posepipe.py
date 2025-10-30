#!/usr/bin/env python3
"""
Method 3: PosePipe (MediaPipe Pose Landmarker, multi-person)
- Same CLI as other methods: <in.mp4> <out.json> <vis_dir> <n_overlay> <out.mp4>
- Outputs your schema: frames -> people -> kpts [[x,y,conf] x 17]
- Keeps only the most-moving person (walker) via motion filter
"""
import sys, os, json, math, cv2, numpy as np
from collections import defaultdict
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

COCO17 = ["nose","left_eye","right_eye","left_ear","right_ear",
          "left_shoulder","right_shoulder","left_elbow","right_elbow",
          "left_wrist","right_wrist","left_hip","right_hip",
          "left_knee","right_knee","left_ankle","right_ankle"]
# MediaPipe indices that map well to COCO
MP = {
  "nose":0, "left_eye":2, "right_eye":5, "left_ear":7, "right_ear":8,
  "left_shoulder":11, "right_shoulder":12, "left_elbow":13, "right_elbow":14,
  "left_wrist":15, "right_wrist":16, "left_hip":23, "right_hip":24,
  "left_knee":25, "right_knee":26, "left_ankle":27, "right_ankle":28
}
COCO2MP = [MP[n] for n in COCO17]
L_ANK, R_ANK = 15, 16

# --- tiny EMA to steady joints across frames ---
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

def main():
    if len(sys.argv)!=6:
        print("Usage: method3_posepipe.py <in.mp4> <out.json> <vis_dir> <n_overlay> <out.mp4>")
        sys.exit(1)
    in_mp4, out_json, vis_dir, n_overlay_str, out_mp4 = sys.argv[1:]
    n_overlay = int(n_overlay_str)

    base_opts = mp_python.BaseOptions(model_asset_path="models/pose_landmarker_full.task")
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=2,  # walker + bystander
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {in_mp4}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if n_overlay>0: os.makedirs(vis_dir, exist_ok=True)
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), max(fps,1.0), (W,H))

    frames=[]
    fidx=0; saved=0
    # EMA per person slot (0/1) per joint (17) for smoothing
    ema = [[EMA(0.25) for _ in range(17)] for _ in range(2)]
    # track motion per slot (0/1) using ankle midpoints
    path = defaultdict(float)
    prev_cent = {0:None, 1:None}

    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        ts_ms = int((fidx/(fps if fps>0 else 1))*1000)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_vision.Image(image_format=mp_vision.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        people=[]
        canvas = np.zeros_like(frame_bgr)

        if result and result.pose_landmarks:
            # Up to 2 persons; stable order is not guaranteed, but we just need "most-moving"
            for slot, lms in enumerate(result.pose_landmarks[:2]):
                kpts=[]
                for j, mp_idx in enumerate(COCO2MP):
                    if mp_idx >= len(lms):
                        kpts.append([None,None,None]); continue
                    lm = lms[mp_idx]
                    x = None if lm.x is None else lm.x*W
                    y = None if lm.y is None else lm.y*H
                    c = getattr(lm,"visibility", None)
                    # EMA smoothing (x,y only if present)
                    if x is not None and y is not None:
                        x = ema[slot][j].step(x)
                        y = ema[slot][j].step(y)
                    kpts.append([x,y,c])
                people.append({"kpts": kpts})
                draw_skeleton(canvas, kpts)

                # update path length for this slot
                c = ankle_mid(kpts)
                if c is not None:
                    if prev_cent[slot] is not None:
                        dx=c[0]-prev_cent[slot][0]; dy=c[1]-prev_cent[slot][1]
                        path[slot]+=math.hypot(dx,dy)
                    prev_cent[slot]=c

        vw.write(canvas)
        if saved < n_overlay:
            cv2.imwrite(os.path.join(vis_dir, f"{fidx:06d}.jpg"), canvas)
            saved += 1

        frames.append({"frame_idx": fidx, "ms": ts_ms, "people": people})
        fidx += 1

    cap.release(); vw.release()

    # Keep only the most-moving slot (walker); rebuild frames
    keep_slot = None
    if path:
        keep_slot = max(path, key=path.get)
    filtered=[]
    for fr in frames:
        ppl = fr["people"]
        if keep_slot is None or not ppl:
            filtered.append({"frame_idx": fr["frame_idx"], "ms": fr["ms"], "people": ppl})
        else:
            kept = [ppl[keep_slot]] if len(ppl)>keep_slot else []
            filtered.append({"frame_idx": fr["frame_idx"], "ms": fr["ms"], "people": kept})

    out = {"video": in_mp4, "fps": fps, "width": W, "height": H, "frames": filtered}
    with open(out_json, "w") as f:
        json.dump(out, f)

    print(f"Saved keypoints JSON -> {out_json}")
    if n_overlay>0:
        print(f"Saved overlays -> {vis_dir} (first {n_overlay} frames)")
    print(f"Saved skeleton-only MP4 -> {out_mp4}")

if __name__ == "__main__":
    main()
