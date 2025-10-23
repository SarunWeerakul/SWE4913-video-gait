import sys, json, pathlib, cv2
from ultralytics import YOLO

video_path = pathlib.Path(sys.argv[1])
json_out   = pathlib.Path(sys.argv[2])
overlay_dir= pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
n_overlay  = int(sys.argv[4]) if len(sys.argv) > 4 else 10
mp4_out    = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else None

model = YOLO("yolov8n-pose.pt")  # downloads on first run

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"Could not open {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

if overlay_dir:
    overlay_dir.mkdir(parents=True, exist_ok=True)

writer = None
if mp4_out:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_out), fourcc, fps, (W, H))

out = {"video": str(video_path), "fps": fps, "width": W, "height": H, "frames": []}

idx = -1
ok, frame = cap.read()
while ok:
    idx += 1
    ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    results = model(frame, verbose=False)[0]

    people = []
    if results.keypoints is not None and len(results.keypoints) > 0:
        n = len(results.keypoints)
        xy = results.keypoints.xy.cpu().numpy()      # (n,k,2)
        conf = getattr(results.keypoints, "conf", None)
        conf = conf.cpu().numpy() if conf is not None else None  # (n,k) or None
        for i in range(n):
            kpts = []
            for j, (x, y) in enumerate(xy[i]):
                cj = float(conf[i, j]) if conf is not None else None
                kpts.append([float(x), float(y), cj])  # [x,y,conf]
            people.append({"kpts": kpts})

    out["frames"].append({"frame_idx": idx, "ms": ts_ms, "people": people})

    ann = results.plot()
    if overlay_dir and idx < n_overlay:
        cv2.imwrite(str(overlay_dir / f"yolo_{idx:04d}.png"), ann)
    if writer and idx < n_overlay:
        writer.write(ann)

    ok, frame = cap.read()

cap.release()
if writer: writer.release()

json_out.parent.mkdir(parents=True, exist_ok=True)
with open(json_out, "w") as f:
    json.dump(out, f)
print(f"Saved keypoints JSON -> {json_out}")
if overlay_dir: print(f"Saved overlays -> {overlay_dir} (first {n_overlay} frames)")
if mp4_out:     print(f"Saved short MP4 -> {mp4_out}")
