import json
import csv
from pathlib import Path

# folder that contains your keypoints.json files
INPUT_DIR = Path(".")   # change this if needed

def json_to_csv(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    video_name = data.get("video", json_path.stem)

    # Find first person to infer number of keypoints
    first_person = None
    for fr in frames:
        people = fr.get("people", [])
        if people:
            first_person = people[0]
            break

    # If no people at all, nothing to write
    if not first_person:
        print(f"No people found in {json_path.name}, skipping.")
        return

    kpts_example = first_person.get("kpts", [])
    n_kpts = len(kpts_example) // 3  # (x, y, c) triplets

    # Build header
    header = [
        "video",
        "frame_idx",
        "ms",
        "person_idx",  # index within this frame
        "tid",         # tracking id if available
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    for i in range(n_kpts):
        header.extend([f"kpt_{i}_x", f"kpt_{i}_y", f"kpt_{i}_c"])

    csv_path = json_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for fr in frames:
            frame_idx = fr.get("frame_idx")
            ms = fr.get("ms")
            people = fr.get("people", [])

            for person_idx, person in enumerate(people):
                kpts = person.get("kpts", [])

                # Ensure length matches expected (n_kpts * 3)
                if len(kpts) != n_kpts * 3:
                    # You can decide to skip or pad; here we skip inconsistent data
                    continue

                tid = person.get("tid")
                bbox = person.get("bbox") or [None, None, None, None]
                # Make sure bbox has 4 elements
                bbox = (bbox + [None] * 4)[:4]

                row = [
                    video_name,
                    frame_idx,
                    ms,
                    person_idx,
                    tid,
                    *bbox,
                ]

                # Flatten keypoints
                for i in range(n_kpts):
                    x = kpts[3 * i]
                    y = kpts[3 * i + 1]
                    c = kpts[3 * i + 2]
                    row.extend([x, y, c])

                writer.writerow(row)

    print(f"Wrote {csv_path}")

if __name__ == "__main__":
    for json_path in INPUT_DIR.glob("*.json"):
        json_to_csv(json_path)
