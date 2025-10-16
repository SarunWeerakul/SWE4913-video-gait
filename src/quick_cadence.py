import sys, csv, numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else "tmp/pose_landmarks.csv"
L_ANKLE, R_ANKLE = 27, 28

frames, ms, ly, ry, lvis, rvis = [], [], [], [], [], []
with open(csv_path) as f:
    r = csv.DictReader(f)
    for row in r:
        frames.append(int(row["frame_idx"]))
        ms.append(int(row["ms"]))
        def get(i, k):
            v = row.get(f"lmk{i}_{k}", "")
            return float(v) if v not in ("", None) else np.nan
        ly.append(get(L_ANKLE, "y"))
        ry.append(get(R_ANKLE, "y"))
        lvis.append(get(L_ANKLE, "vis"))
        rvis.append(get(R_ANKLE, "vis"))

frames = np.array(frames); ms = np.array(ms)
ly, ry = np.array(ly), np.array(ry)
lvis, rvis = np.array(lvis), np.array(rvis)

mask = (lvis > 0.5) & (rvis > 0.5)
t = (ms[mask] - ms[mask][0]) / 1000.0
sig = (ly[mask] + ry[mask]) / 2.0

if t.size < 10:
    print("Not enough confident points.")
    raise SystemExit(0)

sig = sig - np.nanmean(sig)
win = max(5, int(0.25 / np.median(np.diff(t))))
sig_sm = np.convolve(sig, np.ones(win)/win, mode="same")

d = np.diff(sig_sm, prepend=sig_sm[0])
zc = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
keep = []
last_t = -1e9
for idx in zc:
    if t[idx] - last_t >= 0.3:
        keep.append(idx); last_t = t[idx]
peaks_t = t[keep]

if len(peaks_t) >= 2:
    step_intervals = np.diff(peaks_t)
    cadence_spm = 60.0 / np.median(step_intervals)
    print(f"Estimated cadence: {cadence_spm:.1f} steps/min")
    print(f"Median step interval: {np.median(step_intervals):.2f}s")
else:
    print("Could not detect steps reliably.")
