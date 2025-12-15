"""
Visualize alignment between two cameras using shank angle difference.

We assume input CSVs come from extract_shank_angle_signal.py and contain:

  frame_idx, ms, angle_L_deg, angle_R_deg, angle_diff_deg, ...

We:
  - load time (ms) and angle_diff_deg from each file
  - resample to a common, uniform time grid
  - compute lag via cross-correlation
  - plot:
      (1) raw signals vs time
      (2) csv2 shifted by the detected lag, overlaid with csv1

Usage (from project root):

python src/tools/plot_shank_alignment.py \
  --csv1 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/shank_angle.csv \
  --csv2 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam2/shank_angle.csv \
  --label1 cam1 \
  --label2 cam2 \
  --tmin 15000 \
  --tmax 21000

If --tmin/--tmax are omitted, the full time range is shown.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------- helpers to load CSV ----------

def load_angle_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load shank_angle.csv and return:
      times_ms: shape (N,)
      signal:   shape (N,) angle_diff_deg
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

            v_str = row.get("angle_diff_deg", "")
            try:
                v = float(v_str)
            except ValueError:
                continue

            if math.isfinite(v):
                times.append(t)
                vals.append(v)

    if not times:
        raise ValueError(f"No valid angle_diff_deg data in {path}")

    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


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


# ---------- main plotting logic ----------

def visualize_pair(
    csv1: str | Path,
    csv2: str | Path,
    label1: str,
    label2: str,
    tmin: float | None,
    tmax: float | None,
) -> None:
    print(f"[plot_shank_alignment] csv1 = {csv1}")
    print(f"[plot_shank_alignment] csv2 = {csv2}")
    if tmin is not None or tmax is not None:
        print(f"[plot_shank_alignment] zoom window = [{tmin}, {tmax}] ms")

    t1, y1 = load_angle_csv(csv1)
    t2, y2 = load_angle_csv(csv2)

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

    # Prepare shifted version of y2 for plotting
    y2_shifted = np.empty_like(y2g)
    y2_shifted[:] = np.nan

    if lag_samples >= 0:
        # y2 delayed vs y1: shift LEFT to align
        if lag_samples == 0:
            y2_shifted[:] = y2g
        else:
            y2_shifted[:-lag_samples] = y2g[lag_samples:]
    else:
        # y2 ahead vs y1: shift RIGHT to align
        shift = -lag_samples
        y2_shifted[shift:] = y2g[:-shift]

    # ---------- make plots ----------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: raw signals
    axes[0].plot(t_grid, y1g, label=f"{label1} angle_diff_deg")
    axes[0].plot(t_grid, y2g, label=f"{label2} angle_diff_deg", alpha=0.7)
    axes[0].set_ylabel("angle_diff (deg)")
    axes[0].set_title("Raw shank angle difference vs time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: csv2 shifted by lag
    axes[1].plot(t_grid, y1g, label=f"{label1} (reference)")
    axes[1].plot(
        t_grid,
        y2_shifted,
        label=f"{label2} shifted by {lag_ms:.1f} ms",
        alpha=0.7,
    )
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("angle_diff (deg, aligned)")
    axes[1].set_title("Shank angle difference with csv2 shifted to align with csv1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Optional zoom into a specific time window (in ms)
    if tmin is not None or tmax is not None:
        xmin = tmin if tmin is not None else t_grid.min()
        xmax = tmax if tmax is not None else t_grid.max()
        axes[0].set_xlim(xmin, xmax)
        axes[1].set_xlim(xmin, xmax)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot and visualize alignment between two shank angle signals."
    )
    parser.add_argument("--csv1", required=True, help="shank_angle.csv for camera 1")
    parser.add_argument("--csv2", required=True, help="shank_angle.csv for camera 2")
    parser.add_argument("--label1", default="cam1", help="Label for camera 1")
    parser.add_argument("--label2", default="cam2", help="Label for camera 2")
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Optional start time (ms) for x-axis zoom",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Optional end time (ms) for x-axis zoom",
    )

    args = parser.parse_args()
    visualize_pair(
        args.csv1,
        args.csv2,
        args.label1,
        args.label2,
        args.tmin,
        args.tmax,
    )


if __name__ == "__main__":
    main()
