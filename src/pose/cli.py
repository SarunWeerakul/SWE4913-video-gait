import argparse, subprocess, sys, pathlib
def main():
    ap = argparse.ArgumentParser(prog="pose", description="Pose pipelines")
    ap.add_argument("--method", choices=["yolo"], default="yolo")
    ap.add_argument("in_mp4"); ap.add_argument("out_json"); ap.add_argument("vis_dir")
    ap.add_argument("n_overlay", type=int); ap.add_argument("out_mp4")
    args = ap.parse_args()
    script = pathlib.Path(__file__).resolve().parents[1] / "pipelines" / "method1_yolo.py"
    cmd = [sys.executable, str(script), args.in_mp4, args.out_json, args.vis_dir, str(args.n_overlay), args.out_mp4]
    sys.exit(subprocess.call(cmd))
if __name__ == "__main__": main()
