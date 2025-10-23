import sys, pathlib, cv2, numpy as np, mediapipe as mp

# Args
in_path    = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.mp4")
out_png    = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/gei_hipcenter.png")
max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 600       # how many frames to average
box_size   = int(sys.argv[4]) if len(sys.argv) > 4 else 320       # crop box width=height (pixels)
ema_alpha  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85    # hip center smoothing (0..1)

# Init
cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened(): raise SystemExit(f"Could not open {in_path}")

mp_pose = mp.solutions.pose
mp_seg  = mp.solutions.selfie_segmentation

# Helpers
def clamp(v, lo, hi): return max(lo, min(hi, v))

acc  = None
count= 0
Hc = Wc = box_size
hip_px_ema = None  # (x,y) EMA

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_seg.SelfieSegmentation(model_selection=1) as seg:

    ok, frame = cap.read()
    while ok and count < max_frames:
        H, W = frame.shape[:2]

        # 1) Pose → hip center (pixels)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res_pose = pose.process(rgb)
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks.landmark
            # MediaPipe indices: L hip=23, R hip=24; coords in [0,1]
            hx = (lm[23].x + lm[24].x) * 0.5 * W
            hy = (lm[23].y + lm[24].y) * 0.5 * H
            if hip_px_ema is None:
                hip_px_ema = (hx, hy)
            else:
                hip_px_ema = (
                    ema_alpha * hip_px_ema[0] + (1-ema_alpha) * hx,
                    ema_alpha * hip_px_ema[1] + (1-ema_alpha) * hy
                )
        # fallback: if no pose this frame, keep last EMA (or center)
        if hip_px_ema is None:
            hip_px_ema = (W/2.0, H/2.0)

        cx, cy = hip_px_ema
        half = box_size // 2
        x0 = int(round(cx - half)); y0 = int(round(cy - half))
        x1 = x0 + box_size;         y1 = y0 + box_size

        # clamp to image bounds (keep exact box_size by shifting window)
        if x0 < 0: x1 -= x0; x0 = 0
        if y0 < 0: y1 -= y0; y0 = 0
        if x1 > W: x0 -= (x1 - W); x1 = W
        if y1 > H: y0 -= (y1 - H); y1 = H
        x0 = clamp(x0, 0, W-1); y0 = clamp(y0, 0, H-1)
        x1 = clamp(x1, 1, W);   y1 = clamp(y1, 1, H)

        crop = frame[y0:y1, x0:x1]

        # 2) Segmentation mask on the crop
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res_seg  = seg.process(rgb_crop)
        mask = (res_seg.segmentation_mask > 0.5).astype("float32")  # 0..1

        # 3) Accumulate
        if acc is None:
            acc = np.zeros((Hc, Wc), np.float32)
        # Ensure crop is exactly box_size (it should be after clamping)
        ch, cw = mask.shape
        if ch != Hc or cw != Wc:
            mask = cv2.resize(mask, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        acc += mask
        count += 1

        ok, frame = cap.read()

cap.release()
if count == 0: raise SystemExit("No frames processed.")

gei = (acc / count) * 255.0
gei = gei.clip(0,255).astype("uint8")
out_png.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_png), gei)
print(f"GEI (hip-centered) saved -> {out_png} | frames used: {count} | box: {box_size}px")
