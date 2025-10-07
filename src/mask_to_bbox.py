import sys, csv, re, pathlib
import cv2
import numpy as np

frames_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/frames")
out_csv = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "tmp/bboxes.csv")
frames = sorted(frames_dir.glob("frame_*.png"))
masks  = sorted(frames_dir.glob("mask_*.png"))

def frame_index(p):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1

frames.sort(key=frame_index)
masks.sort(key=frame_index)

rows = [("frame_idx","x","y","w","h","area","cx","cy")]
for f,m in zip(frames, masks):
    mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    # threshold just in case
    _,thr = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(thr > 0)
    if xs.size == 0 or ys.size == 0:
        rows.append((frame_index(f), -1,-1,0,0,0,-1,-1))
        continue
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    area = int((thr > 0).sum())
    cx, cy = float(xs.mean()), float(ys.mean())
    rows.append((frame_index(f), x0, y0, w, h, area, cx, cy))

out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"Wrote {len(rows)-1} rows → {out_csv}")
