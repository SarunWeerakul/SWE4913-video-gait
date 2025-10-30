import argparse, subprocess, sys, pathlib
def main():
    ap = argparse.ArgumentParser(prog="gei", description="GEI pipelines")
    ap.add_argument("--method", choices=["hipcenter"], default="hipcenter")
    ap.add_argument("in_mp4"); ap.add_argument("out_png")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()
    script = pathlib.Path(__file__).resolve().parents[1] / "methods" / "hipcenter.py"
    cmd = [sys.executable, str(script), args.in_mp4, args.out_png, str(args.start), str(args.frames)]
    sys.exit(subprocess.call(cmd))
if __name__ == "__main__": main()
