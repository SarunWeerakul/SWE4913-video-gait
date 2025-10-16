import sys, pathlib, cv2, numpy as np, mediapipe as mp

in_path  = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.mp4")
out_png  = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/gei.png")
max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 300  # safety cap

cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened():
    raise SystemExit(f"Could not open {in_path}")

ok, first = cap.read()
if not ok:
    raise SystemExit("No frames read.")
H, W = first.shape[:2]

acc = np.zeros((H, W), np.float32)
count = 0

with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
    frame = first
    while ok and count < max_frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        mask = (res.segmentation_mask > 0.5).astype("float32")  # 0/1
        acc += mask
        count += 1
        ok, frame = cap.read()

cap.release()
if count == 0:
    raise SystemExit("No frames processed.")

gei = (acc / count) * 255.0
gei = gei.clip(0,255).astype("uint8")
out_png.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_png), gei)
print(f"GEI saved → {out_png} using {count} frames")
