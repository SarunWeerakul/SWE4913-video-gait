# src/pose/cli.py
import argparse
from importlib import import_module

def main():
    ap = argparse.ArgumentParser("pose")
    ap.add_argument("--method",
                    choices=["method1_yolo", "method2_topdown", "method3_posepipe"],
                    default="method1_yolo")
    ap.add_argument("in_mp4")
    ap.add_argument("out_json")
    ap.add_argument("vis_dir")
    ap.add_argument("n_overlay", type=int)
    ap.add_argument("out_mp4")
    args = ap.parse_args()

    mod = import_module(f"pose.pipelines.{args.method}")  # <— canonical import
    if hasattr(mod, "main"):
        mod.main(args.in_mp4, args.out_json, args.vis_dir, args.n_overlay, args.out_mp4)
    else:
        raise SystemExit(f"Pipeline {args.method} has no main()")

if __name__ == "__main__":
    main()
