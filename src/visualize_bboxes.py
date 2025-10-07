import sys, csv, pathlib, cv2

frames_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/frames")
csv_path   = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/bboxes.csv")
out_dir    = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "tmp/vis")
out_dir.mkdir(parents=True, exist_ok=True)

rows = {}
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        rows[int(row["frame_idx"])] = row

for frame_path in sorted(frames_dir.glob("frame_*.png")):
    idx = int(frame_path.stem.split("_")[-1])
    img = cv2.imread(str(frame_path))
    row = rows.get(idx)
    if row and int(row["w"]) > 0:
        x,y,w,h = map(int, (row["x"], row["y"], row["w"], row["h"]))
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cx,cy = int(float(row["cx"])), int(float(row["cy"]))
        cv2.circle(img, (cx,cy), 3, (255,0,0), -1)
    cv2.imwrite(str(out_dir / f"vis_{idx:04d}.png"), img)

print(f"Wrote visualizations to {out_dir}")
