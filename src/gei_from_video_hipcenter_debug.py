import sys, pathlib, cv2, numpy as np, mediapipe as mp
in_path    = pathlib.Path(sys.argv[1])
out_png    = pathlib.Path(sys.argv[2])
max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 600
box_size   = int(sys.argv[4]) if len(sys.argv) > 4 else 320
ema_alpha  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85
debug_mp4  = pathlib.Path(sys.argv[6]) if len(sys.argv) > 6 else None

cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened(): raise SystemExit(f"Could not open {in_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = None
if debug_mp4:
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(debug_mp4), fourcc, fps, (W, H))

mp_pose = mp.solutions.pose
mp_seg  = mp.solutions.selfie_segmentation

def clamp(v, lo, hi): return max(lo, min(hi, v))
acc = np.zeros((box_size, box_size), np.float32)
count=0
hip_px_ema=None

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_seg.SelfieSegmentation(model_selection=1) as seg:

  ok, frame = cap.read()
  while ok and count < max_frames:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_pose = pose.process(rgb)
    if res_pose.pose_landmarks:
      lm = res_pose.pose_landmarks.landmark
      hx = (lm[23].x + lm[24].x) * 0.5 * W
      hy = (lm[23].y + lm[24].y) * 0.5 * H
      hip_px_ema = (hx, hy) if hip_px_ema is None else (
        ema_alpha*hip_px_ema[0] + (1-ema_alpha)*hx,
        ema_alpha*hip_px_ema[1] + (1-ema_alpha)*hy
      )
    if hip_px_ema is None:
      hip_px_ema = (W/2.0, H/2.0)

    cx, cy = hip_px_ema
    half = box_size//2
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = x0 + box_size; y1 = y0 + box_size
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > W: x0 -= (x1 - W); x1 = W
    if y1 > H: y0 -= (y1 - H); y1 = H
    x0 = clamp(x0,0,W-1); y0 = clamp(y0,0,H-1)
    x1 = clamp(x1,1,W);   y1 = clamp(y1,1,H)

    crop = frame[y0:y1, x0:x1]
    res_seg = seg.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    mask = (res_seg.segmentation_mask > 0.5).astype("float32")
    ch,cw = mask.shape
    if (ch,cw)!=(box_size,box_size):
      mask = cv2.resize(mask, (box_size,box_size), interpolation=cv2.INTER_NEAREST)
    acc += mask; count += 1

    # draw debug
    dbg = frame.copy()
    cv2.rectangle(dbg, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.circle(dbg, (int(cx),int(cy)), 4, (255,0,0), -1)
    if writer: writer.write(dbg)

    ok, frame = cap.read()

cap.release()
if writer: writer.release()
if count==0: raise SystemExit("No frames processed.")

gei = (acc / count * 255.0).clip(0,255).astype("uint8")
out_png.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_png), gei)
print(f"Saved GEI -> {out_png} | frames={count} | box={box_size}px | debug={'yes' if writer else 'no'}")
