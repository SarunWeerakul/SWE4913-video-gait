import argparse, sys, inspect
from importlib import import_module

def main():
    ap = argparse.ArgumentParser("pose")
    ap.add_argument(
        "--method",
        choices=["method1_yolo", "method2_topdown", "method3_posepipe"],
        default="method1_yolo",
    )
    ap.add_argument("in_mp4")
    ap.add_argument("out_json")
    ap.add_argument("vis_dir")
    ap.add_argument("n_overlay", type=int)
    ap.add_argument("out_mp4")
    args = ap.parse_args()

    # Map friendly names to real modules (update when you have a real YOLO)
    method_map = {
        "method1_yolo": "method3_posepipe",  # TODO: switch to "method1_yolo" when implemented
        "method2_topdown": "method2_topdown",
        "method3_posepipe": "method3_posepipe",
    }
    target = method_map[args.method]
    mod = import_module(f"pose.pipelines.{target}")

    # Try common entry points in a robust way
    if hasattr(mod, "main"):
        sig = inspect.signature(mod.main)
        if len(sig.parameters) == 0:
            # Module expects CLI-style argv; build it and call main()
            argv = [
                target,
                args.in_mp4,
                args.out_json,
                args.vis_dir,
                str(args.n_overlay),
                args.out_mp4,
            ]
            old_argv = sys.argv[:]
            try:
                sys.argv = argv
                return mod.main()
            finally:
                sys.argv = old_argv
        else:
            # Callable with explicit args
            return mod.main(args.in_mp4, args.out_json, args.vis_dir, args.n_overlay, args.out_mp4)

    if hasattr(mod, "Pipeline"):
        pipe = mod.Pipeline()
        if hasattr(pipe, "run"):
            return pipe.run(args.in_mp4, args.out_json, args.vis_dir, args.n_overlay, args.out_mp4)

    raise SystemExit(f"❌ No usable entry point found in pose.pipelines.{target}")

if __name__ == "__main__":
    main()
