import sys, pathlib
import cv2, mediapipe as mp

in_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.mov"
out_dir = pathlib.Path("tmp/frames"); out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(in_path)
if not cap.isOpened():
    raise SystemExit(f"Could not open {in_path}. Put a short clip there and re-run.")

with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
    ok, frame = cap.read()
    count = 0
    while ok and count < 30:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg.process(rgb)
        mask = (res.segmentation_mask > 0.5).astype("uint8") * 255
        cv2.imwrite(str(out_dir / f"frame_{count:04d}.png"), frame)
        cv2.imwrite(str(out_dir / f"mask_{count:04d}.png"), mask)
        ok, frame = cap.read()
        count += 1

cap.release()
print(f"Wrote {count} frames and masks to {out_dir}")
