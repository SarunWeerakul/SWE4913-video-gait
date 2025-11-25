"""
Visualize the lag between two ankle motion signals from motion_ankle.csv,
with support for starting the plot after a given time (--tmin, in seconds).

We use:
    ankle_rel_norm = (ankle_y - hip_y) / body_height
"""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------- Load CSV ----------

def load_motion_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load motion_ankle.csv and return (times_ms, ankle_rel_norm)."""
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

    return np.asarray(times, float), np.asarray(vals, float)


# ---------- Resample ----------

def resample_to_common_grid(t1, y1, t2, y2):
    """Resample signals onto a common uniform time grid."""
    t_start = max(t1.min(), t2.min())
    t_end = min(t1.max(), t2.max())
    if t_end <= t_start:
        raise ValueError("No overlapping time range.")

    dt1 = np.median(np.diff(t1))
    dt2 = np.median(np.diff(t2))
    dt = float(np.median([dt1, dt2]))

    n_steps = int((t_end - t_start) / dt) + 1
    t_grid = t_start + dt * np.arange(n_steps)

    y1g = np.interp(t_grid, t1, y1)
    y2g = np.interp(t_grid, t2, y2)
    return t_grid, y1g, y2g


# ---------- Lag Computation ----------

def compute_lag(y1, y2, dt_ms):
    """Compute lag via cross-correlation."""
    y1c = y1 - np.mean(y1)
    y2c = y2 - np.mean(y2)

    corr = np.correlate(y1c, y2c, mode="full")
    n = len(y1c)
    lags = np.arange(-n + 1, n)

    idx = int(np.argmax(corr))
    lag_samples = int(lags[idx])
    lag_ms = lag_samples * dt_ms

    denom = np.linalg.norm(y1c) * np.linalg.norm(y2c)
    max_corr = float(corr[idx] / denom) if denom > 0 else 0.0

    return lag_samples, lag_ms, max_corr


# ---------- Visualization ----------

def visualize_pair(csv1, csv2, tmin_sec=0.0):
    print(f"[plot_motion_alignment] csv1 = {csv1}")
    print(f"[plot_motion_alignment] csv2 = {csv2}")

    t1, y1 = load_motion_csv(csv1)
    t2, y2 = load_motion_csv(csv2)

    # ----- (NEW) Time filtering -----
    if tmin_sec > 0:
        tmin_ms = tmin_sec * 1000
        keep1 = t1 >= tmin_ms
        keep2 = t2 >= tmin_ms
        t1, y1 = t1[keep1], y1[keep1]
        t2, y2 = t2[keep2], y2[keep2]
        print(f"[INFO] Filtering samples before {tmin_sec} sec ({tmin_ms} ms).")

    # Resample both signals
    t_grid, y1g, y2g = resample_to_common_grid(t1, y1, t2, y2)
    dt = np.median(np.diff(t_grid))

    # Lag
    lag_samples, lag_ms, max_corr = compute_lag(y1g, y2g, dt_ms=dt)

    print("--------------------------------------------------")
    print(f"tmin (s)             : {tmin_sec}")
    print(f"Common dt (ms)       : {dt:.3f}")
    print(f"Lag (samples)        : {lag_samples}")
    print(f"Lag (ms)             : {lag_ms:.3f}")
    print(f"Max normalized corr  : {max_corr:.3f}")

    # Shift signal 2
    y2_shift = np.full_like(y2g, np.nan)
    if lag_samples >= 0:
        if lag_samples == 0:
            y2_shift[:] = y2g
        else:
            y2_shift[:-lag_samples] = y2g[lag_samples:]
    else:
        shift = -lag_samples
        y2_shift[shift:] = y2g[:-shift]

    # ----- PLOT -----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_grid, y1g, label="csv1 ankle_rel_norm")
    axes[0].plot(t_grid, y2g, label="csv2 ankle_rel_norm", alpha=0.7)
    axes[0].set_ylabel("ankle_rel_norm")
    axes[0].set_title("Raw ankle_rel_norm vs time (ms)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_grid, y1g, label="csv1 (reference)")
    axes[1].plot(t_grid, y2_shift, label=f"csv2 shifted by {lag_ms:.1f} ms", alpha=0.7)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("aligned ankle_rel_norm")
    axes[1].set_title("csv2 shifted to align with csv1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Plot motion alignment between two ankle signals with optional tmin."
    )
    parser.add_argument("--csv1", required=True, help="First motion_ankle.csv")
    parser.add_argument("--csv2", required=True, help="Second motion_ankle.csv")

    parser.add_argument(
        "--tmin",
        type=float,
        default=0.0,
        help="Start plotting after this many seconds (default: 0)",
    )

    args = parser.parse_args()
    visualize_pair(args.csv1, args.csv2, tmin_sec=args.tmin)


if __name__ == "__main__":
    main()
