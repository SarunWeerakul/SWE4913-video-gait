"""
Find the matching frame in camera 2 for a given frame in camera 1,
using the relative ankle signal (ankle_rel_norm) to estimate the lag.

Steps:
  1) Load motion_ankle.csv for cam1 and cam2.
  2) Estimate lag_ms between them (same logic as align_motion_signals.py).
  3) Take a reference frame_idx from cam1, get its ms time.
  4) target_ms_cam2 = ms_cam1 + lag_ms
  5) Find frame in cam2 whose ms is closest to target_ms_cam2.
  6) Print both frame indices and suggested image filenames.

Usage (from project root):

python src/tools/find_matching_frame.py \
  --csv1 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/motion_ankle.csv \
  --csv2 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam2/motion_ankle.csv \
  --frame1 1002
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------- load CSV as time series ----------

def load_motion_series(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a motion_ankle.csv file and return:
      times_ms: shape (N,)
      signal:   shape (N,) ankle_rel_norm
    """
    path = Path(path)
    times: List[float] = []
    vals: List[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["ms"])
            except (KeyError, ValueError):
                continue

            v_str = row.get("ankle_rel_norm", "")
            try:
                v = float(v_str)
            except ValueError:
                continue

            if math.isfinite(v):
                times.append(t)
                vals.append(v)

    if not times:
        raise ValueError(f"No valid data in {path}")

    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def load_frame_time_map(path: str | Path) -> Dict[int, float]:
    """
    Load a motion_ankle.csv file and build:
      frame_idx -> ms
    """
    path = Path(path)
    mapping: Dict[int, float] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fi = int(row["frame_idx"])
                t = float(row["ms"])
            except (KeyError, ValueError):
                continue
            mapping[fi] = t
    return mapping


# ---------- resampling + lag estimation (same as align_motion_signals.py) ----------

def resample_to_common_grid(
    t1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample y1 and y2 onto a common, uniform time grid (in ms).
    """
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    if t_end <= t_start:
        raise ValueError("No overlapping time range between the two signals.")

    dt1 = np.median(np.diff(t1))
    dt2 = np.median(np.diff(t2))
    dt = float(np.median([dt1, dt2]))
    if dt <= 0:
        raise ValueError("Non-positive timestep detected in input data.")

    n_steps = int((t_end - t_start) / dt) + 1
    t_grid = t_start + dt * np.arange(n_steps)

    y1_grid = np.interp(t_grid, t1, y1)
    y2_grid = np.interp(t_grid, t2, y2)

    return t_grid, y1_grid, y2_grid


def compute_lag(
    y1: np.ndarray, y2: np.ndarray, dt_ms: float
) -> Tuple[int, float, float]:
    """
    Compute lag between y1 and y2 using cross-correlation.

    Returns:
      lag_samples: int  (positive => y2 is delayed vs y1)
      lag_ms:      float (lag_samples * dt_ms)
      max_corr:    float (value of maximum normalized correlation)
    """
    y1c = y1 - np.mean(y1)
    y2c = y2 - np.mean(y2)

    corr = np.correlate(y1c, y2c, mode="full")

    n = len(y1c)
    lags = np.arange(-n + 1, n)

    idx_max = int(np.argmax(corr))
    lag_samples = int(lags[idx_max])
    lag_ms = lag_samples * dt_ms

    denom = np.linalg.norm(y1c) * np.linalg.norm(y2c)
    if denom > 0:
        max_corr = float(corr[idx_max] / denom)
    else:
        max_corr = 0.0

    return lag_samples, lag_ms, max_corr


# ---------- main logic ----------

def find_matching_frame(csv1: str | Path, csv2: str | Path, frame1: int) -> None:
    csv1 = Path(csv1)
    csv2 = Path(csv2)
    print(f"[find_matching_frame] csv1 = {csv1}")
    print(f"[find_matching_frame] csv2 = {csv2}")
    print(f"[find_matching_frame] reference frame in cam1: {frame1}")

    # 1) load time series for lag estimation
    t1, y1 = load_motion_series(csv1)
    t2, y2 = load_motion_series(csv2)

    t_grid, y1g, y2g = resample_to_common_grid(t1, y1, t2, y2)
    dt = np.median(np.diff(t_grid))
    lag_samples, lag_ms, max_corr = compute_lag(y1g, y2g, dt_ms=dt)

    print("--------------------------------------------------")
    print(f"Estimated lag (samples): {lag_samples}")
    print(f"Estimated lag (ms)     : {lag_ms:.3f}")
    print(f"Max normalized corr    : {max_corr:.3f}")
    print()

    # 2) map frame_idx -> ms
    map1 = load_frame_time_map(csv1)
    map2 = load_frame_time_map(csv2)

    if frame1 not in map1:
        raise ValueError(f"frame_idx {frame1} not found in {csv1}")

    t_ref = map1[frame1]
    # important: same rule as align script:
    # event in cam1 at t_ref ≈ event in cam2 at t_ref + lag_ms
    t_target = t_ref + lag_ms

    # 3) find closest time in cam2
    ms_values_cam2 = np.array(list(map2.values()), dtype=float)
    frame_indices_cam2 = np.array(list(map2.keys()), dtype=int)

    idx_best = int(np.argmin(np.abs(ms_values_cam2 - t_target)))
    frame2 = int(frame_indices_cam2[idx_best])
    t2_best = float(ms_values_cam2[idx_best])
    time_error = t2_best - t_target

    print("--------------------------------------------------")
    print(f"cam1: frame_idx = {frame1}, ms = {t_ref:.3f}")
    print(f"target time in cam2 (ms)      : {t_target:.3f}")
    print(f"closest frame in cam2: idx    : {frame2}")
    print(f"closest frame in cam2: ms     : {t2_best:.3f}")
    print(f"time difference (ms)          : {time_error:.3f}")
    print()

    # suggest filenames (assuming pattern frame_XXXXXX.png)
    fname1 = f"frame_{frame1:06d}.png"
    fname2 = f"frame_{frame2:06d}.png"
    print("Suggested image filenames (if using frame_XXXXXX.png):")
    print(f"  cam1: {fname1}")
    print(f"  cam2: {fname2}")


def main():
    parser = argparse.ArgumentParser(
        description="Find matching frame in camera 2 for a given frame index in camera 1."
    )
    parser.add_argument("--csv1", required=True, help="motion_ankle.csv for camera 1 (reference)")
    parser.add_argument("--csv2", required=True, help="motion_ankle.csv for camera 2")
    parser.add_argument("--frame1", type=int, required=True, help="frame_idx in camera 1 to match")

    args = parser.parse_args()
    find_matching_frame(args.csv1, args.csv2, args.frame1)


if __name__ == "__main__":
    main()
