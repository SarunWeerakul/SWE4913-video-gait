#!/usr/bin/env python3
import argparse, json, csv, math
from pathlib import Path

# COCO-17 indices
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12; L_KNE=13; R_KNE=14; L_ANK=15; R_ANK=16

IDX2NAME = {
    L_HIP:  "left_hip",   R_HIP:  "right_hip",
    L_KNE:  "left_knee",  R_KNE:  "right_knee",
    L_ANK:  "left_ankle", R_ANK:  "right_ankle",
}

def load_json(p: Path):
    with p.open("r") as f:
        return json.load(f)

def get_pt(kpts, idx, min_conf):
    """Return (x,y,conf) or (None,None,None)."""
    if not isinstance(kpts, list) or len(kpts) <= idx:
        return (None, None, None)
    kp = kpts[idx]
    if not isinstance(kp, (list, tuple)) or len(kp) < 2:
        return (None, None, None)
    x = kp[0]; y = kp[1]; c = kp[2] if len(kp) > 2 else None
    if c is not None and c < min_conf:
        return (None, None, None)
    return (x, y, c)

def midpoint(p1, p2):
    if p1[0] is None or p2[0] is None:
        return (None, None)
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)

def euclid(a, b):
    if a[0] is None or b[0] is None:
        return float("inf")
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return math.hypot(dx, dy)

def extract_people(fr, min_conf):
    """Yield dicts: {frame_idx, ms, person_index, kpts, ref_point(x,y)}."""
    frame_idx = fr.get("frame_idx"); ms = fr.get("ms")
    people = fr.get("people", []) or []
    for pidx, person in enumerate(people):
        kpts = person.get("kpts", []) or []
        # ref point = ankle midpoint (fallback to hip midpoint)
        la = get_pt(kpts, L_ANK, min_conf)
        ra = get_pt(kpts, R_ANK, min_conf)
        ref = midpoint(la, ra)
        if ref[0] is None:
            lh = get_pt(kpts, L_HIP, min_conf)
            rh = get_pt(kpts, R_HIP, min_conf)
            ref = midpoint(lh, rh)
        yield {
            "frame_idx": frame_idx, "ms": ms,
            "person_index": pidx, "kpts": kpts, "ref": ref
        }

def link_tracks(frames, min_conf=0.3, max_link=60.0):
    """
    Very simple tracker:
      - For each frame, assign each person to the nearest existing track
        within max_link pixels (by ref point). Otherwise create new track.
    Returns tracks: dict track_id -> list of observations.
    """
    next_tid = 1
    tracks = {}              # tid -> list of obs
    last_pos = {}            # tid -> (x,y)

    for fr in frames:
        obs = list(extract_people(fr, min_conf))
        # greedy assignment: for each obs, find nearest track
        assigned = set()
        used_tids = set()
        for o in obs:
            # find best track
            best_tid, best_d = None, float("inf")
            for tid, pos in last_pos.items():
                d = euclid(o["ref"], pos)
                if d < best_d:
                    best_tid, best_d = tid, d
            if best_d <= max_link:
                # assign to existing track
                tracks.setdefault(best_tid, []).append({**o, "track_id": best_tid})
                last_pos[best_tid] = o["ref"]
                used_tids.add(best_tid)
                assigned.add(id(o))

        # new tracks for unassigned
        for o in obs:
            if id(o) in assigned:
                continue
            tid = next_tid; next_tid += 1
            tracks.setdefault(tid, []).append({**o, "track_id": tid})
            last_pos[tid] = o["ref"]
            used_tids.add(tid)

        # (optional) could prune dead tracks; not needed here

    return tracks

def path_length(points):
    """Total path length over sequence of (x,y)."""
    total = 0.0
    prev = None
    for pt in points:
        if prev is not None:
            d = euclid(prev, pt)
            if math.isfinite(d):
                total += d
        prev = pt
    return total

def write_csv(rows, out_path: Path):
    cols = (
        ["track_id", "frame_idx", "ms", "person_index"] +
        [f"{n}_{s}" for n in IDX2NAME.values() for s in ("x","y","conf")]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def build_row(obs):
    """Map an observation to one CSV row with hips/knees/ankles."""
    kpts = obs["kpts"]
    row = {
        "track_id": obs["track_id"],
        "frame_idx": obs["frame_idx"],
        "ms": obs["ms"],
        "person_index": obs["person_index"],
    }
    for i, name in IDX2NAME.items():
        x = y = c = None
        if isinstance(kpts, list) and len(kpts) > i and isinstance(kpts[i], (list, tuple)):
            kp = kpts[i]
            x = kp[0] if len(kp) > 0 else None
            y = kp[1] if len(kp) > 1 else None
            c = kp[2] if len(kp) > 2 else None
        row[f"{name}_x"] = x
        row[f"{name}_y"] = y
        row[f"{name}_conf"] = c
    return row

def process_file(src: Path, outdir: Path, min_conf: float, max_link: float, min_move: float):
    data = load_json(src)
    frames = data.get("frames", []) if isinstance(data, dict) else []
    tracks = link_tracks(frames, min_conf=min_conf, max_link=max_link)

    # compute movement per track
    movement = {}
    for tid, obs_list in tracks.items():
        pts = [o["ref"] for o in obs_list]
        movement[tid] = path_length(pts)

    # keep only moving-enough tracks
    keep = {tid for tid, mv in movement.items() if mv >= min_move}

    rows = []
    for tid, obs_list in tracks.items():
        if tid not in keep:
            continue
        for o in obs_list:
            rows.append(build_row(o))

    out = outdir / (src.stem + "_knee_ankle_hips.csv")
    write_csv(rows, out)
    return out, movement

def main():
    ap = argparse.ArgumentParser(
        description="Extract hips/knees/ankles from YOLO-pose JSON, with simple tracking and static-person filtering."
    )
    ap.add_argument("inputs", nargs="+", help="Input JSON files or a directory.")
    ap.add_argument("--glob", default="*.json", help="Glob used inside directories (default: *.json).")
    ap.add_argument("--outdir", default=None, help="Output directory (default: alongside each input).")
    ap.add_argument("--min-conf", type=float, default=0.3, help="Min confidence for joints (default: 0.3).")
    ap.add_argument("--max-link", type=float, default=60.0, help="Max px to link detections into a track (default: 60).")
    ap.add_argument("--min-move", type=float, default=30.0, help="Min total path length in px to keep a track (default: 30).")
    args = ap.parse_args()

    in_paths = []
    for s in args.inputs:
        p = Path(s)
        if p.is_dir():
            in_paths.extend(sorted(p.rglob(args.glob)))
        else:
            in_paths.append(p)

    if not in_paths:
        print("No input files found.")
        return

    for src in in_paths:
        try:
            outdir = Path(args.outdir) if args.outdir else src.parent
            outdir.mkdir(parents=True, exist_ok=True)
            out, movement = process_file(
                src, outdir,
                min_conf=args.min_conf,
                max_link=args.max_link,
                min_move=args.min_move,
            )
            kept = sum(1 for mv in movement.values() if mv >= args.min_move)
            total = len(movement)
            print(f"✓ {src} -> {out}  (kept {kept}/{total} tracks; min_move={args.min_move}px)")
        except Exception as e:
            print(f"✗ {src} -> ERROR: {e}")

if __name__ == "__main__":
    main()
