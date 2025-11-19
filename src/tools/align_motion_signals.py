"""
Compute time offset between two motion signals from motion_ankle.csv.

We use the relative ankle signal:

    ankle_rel_norm = (ankle_y - hip_y) / body_height

which is more invariant to distance to the camera.

Usage:

python src/tools/align_motion_signals.py \
  --csv1 path/to/cam1/motion_ankle.csv \
  --csv2 path/to/cam2/motion_ankle.csv
"""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_motion_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
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


def resample_to_common_grid(
    t1: np.ndarray, y1: np.ndarray,
    t2: np.ndarray, y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def analyze_pair(csv1: str | Path, csv2: str | Path) -> None:
    print(f"[align_motion_signals] csv1 = {csv1}")
    print(f"[align_motion_signals] csv2 = {csv2}")

    t1, y1 = load_motion_csv(csv1)
    t2, y2 = load_motion_csv(csv2)

    t_grid, y1g, y2g = resample_to_common_grid(t1, y1, t2, y2)
    dt = np.median(np.diff(t_grid))

    lag_samples, lag_ms, max_corr = compute_lag(y1g, y2g, dt_ms=dt)

    print("--------------------------------------------------")
    print(f"Common dt (ms)       : {dt:.3f}")
    print(f"Number of samples    : {len(t_grid)}")
    print(f"Lag (samples)        : {lag_samples}")
    print(f"Lag (ms)             : {lag_ms:.3f}")
    print(f"Max normalized corr  : {max_corr:.3f}")
    print()
    if lag_samples > 0:
        print("  -> csv2 is delayed relative to csv1.")
    elif lag_samples < 0:
        print("  -> csv2 is ahead of csv1.")
    else:
        print("  -> No lag detected (signals already aligned).")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate time offset between two ankle relative motion signals."
    )
    parser.add_argument("--csv1", required=True, help="First motion_ankle.csv (reference)")
    parser.add_argument("--csv2", required=True, help="Second motion_ankle.csv")

    args = parser.parse_args()
    analyze_pair(args.csv1, args.csv2)


if __name__ == "__main__":
    main()
