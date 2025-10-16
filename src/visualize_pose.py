import sys, pathlib, cv2, mediapipe as mp

in_path  = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.mp4")
out_dir  = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/pose_vis")
limit    = int(sys.argv[3]) if len(sys.argv) > 3 else 60

out_dir.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(in_path))
if not cap.isOpened():
    raise SystemExit(f"Could not open {in_path}")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    idx, ok = -1, True
    while ok and idx < limit-1:
        ok, frame = cap.read()
        if not ok: break
        idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
            )
        cv2.imwrite(str(out_dir / f"pose_{idx:04d}.png"), frame)

cap.release()
print(f"Wrote overlays to {out_dir}")
