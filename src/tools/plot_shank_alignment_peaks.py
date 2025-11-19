"""
Align and visualize shank/thigh angle difference using PEAK timing only.

We assume each CSV has columns:
  frame_idx, ms, angle_diff_deg, ...

Steps:
  1) load ms and angle_diff_deg from csv1 and csv2
  2) resample to a common, uniform time grid
  3) detect local maxima (peaks) in each signal
  4) build spike trains: 0 everywhere, 1 at peak indices
  5) cross-correlate the spike trains to estimate lag
  6) plot:
       (1) original angle_diff_deg vs time with peak markers
       (2) spike trains with csv2 shifted by detected lag

This way the alignment cares mainly about
  - WHEN peaks occur
  - peak-to-peak spacing (step timing)
and not about peak amplitude.

Usage (from project root):

python src/tools/plot_shank_alignment_peaks.py \
  --csv1 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam1/thigh_angle.csv \
  --csv2 tmp/UNB-HTL-1001/BF/W1/method1_yolo/cam2/thigh_angle.csv \
  --label1 cam1 \
  --label2 cam2
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
    Load angle CSV and return:
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


# ---------- peak detection and spike trains ----------

def detect_peaks(
    y: np.ndarray,
    min_prominence: float = 2.0,
    min_distance: int = 5,
) -> np.ndarray:
    """
    Simple 1D local-max peak detection.

    A point i is a peak if:
      - it's not at the boundary
      - y[i] > y[i-1] and y[i] >= y[i+1]
      - (y[i] - local_min) >= min_prominence
    We also enforce a minimum index distance between peaks
    to avoid detecting tiny fluctuations.

    Returns an array of peak indices.
    """
    n = len(y)
    if n < 3:
        return np.array([], dtype=int)

    # crude local minima estimate (for prominence)
    y_min = np.min(y)
    peaks = []

    last_peak = -min_distance - 1

    for i in range(1, n - 1):
        if i - last_peak < min_distance:
            continue

        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            # simple prominence check
            if (y[i] - y_min) >= min_prominence:
                peaks.append(i)
                last_peak = i

    return np.asarray(peaks, dtype=int)


def build_spike_train(length: int, peak_indices: np.ndarray) -> np.ndarray:
    """
    Build a 0/1 spike train of given length with 1s at peak_indices.
    """
    spikes = np.zeros(length, dtype=float)
    for idx in peak_indices:
        if 0 <= idx < length:
            spikes[idx] = 1.0
    return spikes


# ---------- lag estimation using spike trains ----------

def compute_lag_from_spikes(
    s1: np.ndarray, s2: np.ndarray, dt_ms: float
) -> Tuple[int, float, float]:
    """
    Cross-correlate two spike trains to estimate lag.

    Returns:
      lag_samples: int  (positive => s2 is delayed vs s1)
      lag_ms:      float
      max_corr:    float (normalized)
    """
    if np.all(s1 == 0) or np.all(s2 == 0):
        return 0, 0.0, 0.0

    corr = np.correlate(s1, s2, mode="full")
    n = len(s1)
    lags = np.arange(-n + 1, n)

    idx_max = int(np.argmax(corr))
    lag_samples = int(lags[idx_max])
    lag_ms = lag_samples * dt_ms

    denom = np.linalg.norm(s1) * np.linalg.norm(s2)
    if denom > 0:
        max_corr = float(corr[idx_max] / denom)
    else:
        max_corr = 0.0

    return lag_samples, lag_ms, max_corr


# ---------- main plotting logic ----------

def visualize_pair(csv1: str | Path, csv2: str | Path, label1: str, label2: str) -> None:
    print(f"[plot_shank_alignment_peaks] csv1 = {csv1}")
    print(f"[plot_shank_alignment_peaks] csv2 = {csv2}")

    # Load raw angle-diff signals
    t1, y1 = load_angle_csv(csv1)
    t2, y2 = load_angle_csv(csv2)

    # Put them on common time grid
    t_grid, y1g, y2g = resample_to_common_grid(t1, y1, t2, y2)
    dt = np.median(np.diff(t_grid))

    # Detect peaks in each (on the resampled signals)
    # You can tune min_prominence and min_distance as needed.
    peak_idx1 = detect_peaks(y1g, min_prominence=2.0, min_distance=5)
    peak_idx2 = detect_peaks(y2g, min_prominence=2.0, min_distance=5)

    # Build spike trains
    s1 = build_spike_train(len(y1g), peak_idx1)
    s2 = build_spike_train(len(y2g), peak_idx2)

    # Estimate lag from peak timing
    lag_samples, lag_ms, max_corr = compute_lag_from_spikes(s1, s2, dt_ms=dt)

    print("--------------------------------------------------")
    print(f"Common dt (ms)          : {dt:.3f}")
    print(f"Number of samples       : {len(t_grid)}")
    print(f"Number of peaks csv1    : {len(peak_idx1)}")
    print(f"Number of peaks csv2    : {len(peak_idx2)}")
    print(f"Lag (samples, peaks)    : {lag_samples}")
    print(f"Lag (ms, peaks)         : {lag_ms:.3f}")
    print(f"Max norm corr (peaks)   : {max_corr:.3f}")
    print()
    if lag_samples > 0:
        print("  -> csv2 is delayed relative to csv1 (based on peak timing).")
    elif lag_samples < 0:
        print("  -> csv2 is ahead of csv1 (based on peak timing).")
    else:
        print("  -> No lag detected (peaks aligned).")

    # Prepare shifted spike train for plotting
    s2_shifted = np.zeros_like(s2)
    if lag_samples >= 0:
        # csv2 delayed: shift LEFT
        if lag_samples == 0:
            s2_shifted[:] = s2
        else:
            s2_shifted[:-lag_samples] = s2[lag_samples:]
    else:
        # csv2 ahead: shift RIGHT
        shift = -lag_samples
        s2_shifted[shift:] = s2[:-shift]

    # ---------- make plots ----------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: angle_diff with peak markers
    axes[0].plot(t_grid, y1g, label=f"{label1} angle_diff_deg")
    axes[0].plot(t_grid, y2g, label=f"{label2} angle_diff_deg", alpha=0.7)
    axes[0].scatter(
        t_grid[peak_idx1], y1g[peak_idx1],
        marker="o", s=30, label=f"{label1} peaks"
    )
    axes[0].scatter(
        t_grid[peak_idx2], y2g[peak_idx2],
        marker="x", s=40, label=f"{label2} peaks"
    )
    axes[0].set_ylabel("angle_diff (deg)")
    axes[0].set_title("Angle difference with detected peaks")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: spike trains (csv2 shifted)
    axes[1].step(t_grid, s1, where="mid", label=f"{label1} peaks (spikes)")
    axes[1].step(
        t_grid, s2_shifted, where="mid",
        label=f"{label2} peaks shifted by {lag_ms:.1f} ms"
    )
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("spike (0/1)")
    axes[1].set_title("Peak-timing spike trains (csv2 shifted to align with csv1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Align and visualize shank/thigh angle signals using peak timing."
    )
    parser.add_argument("--csv1", required=True, help="angle CSV for camera 1")
    parser.add_argument("--csv2", required=True, help="angle CSV for camera 2")
    parser.add_argument("--label1", default="cam1", help="Label for camera 1")
    parser.add_argument("--label2", default="cam2", help="Label for camera 2")

    args = parser.parse_args()
    visualize_pair(args.csv1, args.csv2, args.label1, args.label2)


if __name__ == "__main__":
    main()
