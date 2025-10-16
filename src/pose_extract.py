# MediaPipe 0.10.x compatible
import sys, csv, pathlib, cv2, mediapipe as mp

in_path   = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.mp4")
out_csv   = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/pose_landmarks.csv")
stride    = int(sys.argv[3]) if len(sys.argv) > 3 else 1
dconf     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
tconf     = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6

cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened():
    raise SystemExit(f"Could not open {in_path}")

mp_pose = mp.solutions.pose
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=dconf,
    min_tracking_confidence=tconf
) as pose, open(out_csv, "w", newline="") as f:

    fieldnames = ["frame_idx","ms"]
    for i in range(33):
        fieldnames += [f"lmk{i}_x", f"lmk{i}_y", f"lmk{i}_z", f"lmk{i}_vis"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    idx = -1
    ok, frame = cap.read()
    while ok:
        idx += 1
        if idx % stride != 0:
            ok, frame = cap.read()
            continue

        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        row = {"frame_idx": idx, "ms": ts_ms}
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                row[f"lmk{i}_x"]   = lm.x
                row[f"lmk{i}_y"]   = lm.y
                row[f"lmk{i}_z"]   = lm.z
                row[f"lmk{i}_vis"] = lm.visibility
        else:
            for i in range(33):
                row[f"lmk{i}_x"]=row[f"lmk{i}_y"]=row[f"lmk{i}_z"]=row[f"lmk{i}_vis"]=""
        writer.writerow(row)
        ok, frame = cap.read()

cap.release()
print(f"Wrote pose landmarks → {out_csv}")
