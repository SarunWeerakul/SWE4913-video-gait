import argparse
import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_angle_csv(path: str | Path):
    """
    Load shank_angle.csv with columns:
      frame_idx, ms, angle_diff_deg (and maybe others)

    Returns:
      times_ms: np.array
      angle_diff_deg: np.array
    """
    path = Path(path)
    times = []
    vals = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["ms"])
            except (KeyError, ValueError):
                continue

            try:
                v = float(row["angle_diff_deg"])
            except (KeyError, ValueError):
                # skip NaNs or missing
                continue

            if math.isfinite(v):
                times.append(t)
                vals.append(v)

    if not times:
        raise ValueError(f"No valid angle_diff_deg data in {path}")

    return np.asarray(times, dtype=float), np.asarray(vals, dtype=float)


def normalize_signal(y: np.ndarray):
    """
    Zero-mean, unit-variance normalization.
    """
    m = np.mean(y)
    s = np.std(y)
    if s <= 0:
        return y * 0.0  # avoid division by zero
    return (y - m) / s


def plot_two_cameras(csv1: str | Path, csv2: str | Path, label1: str, label2: str):
    t1, a1 = load_angle_csv(csv1)
    t2, a2 = load_angle_csv(csv2)

    # Normalize
    a1_norm = normalize_signal(a1)
    a2_norm = normalize_signal(a2)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t1,
            y=a1_norm,
            mode="lines",
            name=f"{label1} (normalized)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t2,
            y=a2_norm,
            mode="lines",
            name=f"{label2} (normalized)",
        )
    )

    fig.update_layout(
        title="Normalized left–right shank angle difference over time",
        xaxis_title="Time (ms)",
        yaxis_title="Normalized angle_diff (z-score)",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot normalized shank angle difference for two cameras using Plotly."
    )
    parser.add_argument(
        "--csv1", required=True, help="shank_angle.csv for camera 1"
    )
    parser.add_argument(
        "--csv2", required=True, help="shank_angle.csv for camera 2"
    )
    parser.add_argument(
        "--label1", default="cam1", help="Label for camera 1"
    )
    parser.add_argument(
        "--label2", default="cam2", help="Label for camera 2"
    )

    args = parser.parse_args()
    plot_two_cameras(args.csv1, args.csv2, args.label1, args.label2)


if __name__ == "__main__":
    main()
