# src/pose/cli.py
import argparse
import inspect
import os
import sys
from importlib import import_module


def _parse_args(argv=None):
    """
    CLI:
      pose --method {yolo,topdown,posepipe,method1_yolo,method2_topdown,method3_posepipe}
           in_mp4 out_json vis_dir n_overlay out_mp4
    """
    ap = argparse.ArgumentParser(prog="pose", description="Pose pipelines runner")
    ap.add_argument(
        "--method",
        choices=[
            "yolo", "topdown", "posepipe",
            "method1_yolo", "method2_topdown", "method3_posepipe",
        ],
        default="method1_yolo",
        help="Pipeline to run",
    )
    ap.add_argument("in_mp4", help="Input video path")
    ap.add_argument("out_json", help="Output JSON path for keypoints/timeline")
    ap.add_argument("vis_dir", help="Directory to write visualization frames")
    ap.add_argument("n_overlay", type=int, help="Max number of frames to dump in vis_dir")
    ap.add_argument("out_mp4", help="Output overlay video path")
    return ap.parse_args(argv)


def _normalize_method(method: str) -> str:
    """
    Accept short aliases and normalize to the canonical method names.
    """
    alias = {
        "yolo": "method1_yolo",
        "topdown": "method2_topdown",
        "posepipe": "method3_posepipe",
    }
    return alias.get(method, method)


def _import_pipeline(target_module: str):
    """
    Import pose.pipelines.<target_module> with a cleaner error message.
    """
    fqmn = f"pose.pipelines.{target_module}"
    try:
        return import_module(fqmn)
    except ModuleNotFoundError as e:
        raise SystemExit(f"❌ Could not import `{fqmn}`. Is the file present and named correctly?\n{e}") from e
    except Exception as e:
        raise SystemExit(f"❌ Error importing `{fqmn}`: {e}") from e


def _ensure_output_paths(out_json: str, vis_dir: str, out_mp4: str):
    """
    Create output directories if they do not exist.
    """
    # vis_dir is a directory; others are files (ensure their parent dirs)
    try:
        os.makedirs(vis_dir, exist_ok=True)
        json_parent = os.path.dirname(out_json) or "."
        mp4_parent = os.path.dirname(out_mp4) or "."
        os.makedirs(json_parent, exist_ok=True)
        os.makedirs(mp4_parent, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"❌ Failed to create output directories: {e}") from e


def _dispatch(mod, in_mp4, out_json, vis_dir, n_overlay, out_mp4):
    """
    Call the pipeline's entry point in a robust way:
      - prefer main(in_mp4, out_json, vis_dir, n_overlay, out_mp4)
      - fall back to main() with argv emulation
      - or Pipeline().run(...)
    """
    if hasattr(mod, "main"):
        sig = inspect.signature(mod.main)
        if len(sig.parameters) == 0:
            # Old-style CLI main(), emulate argv
            argv = [
                mod.__name__.split(".")[-1],
                in_mp4,
                out_json,
                vis_dir,
                str(n_overlay),
                out_mp4,
            ]
            old_argv = sys.argv[:]
            try:
                sys.argv = argv
                return mod.main()
            finally:
                sys.argv = old_argv
        else:
            return mod.main(in_mp4, out_json, vis_dir, n_overlay, out_mp4)

    if hasattr(mod, "Pipeline"):
        pipe = mod.Pipeline()
        if hasattr(pipe, "run"):
            return pipe.run(in_mp4, out_json, vis_dir, n_overlay, out_mp4)

    raise SystemExit(f"❌ No usable entry point found in {mod.__name__} "
                     f"(expected `main(...)`, `main()`, or `Pipeline().run(...)`).")


def main(argv=None):
    args = _parse_args(argv)

    # Normalize/alias the method to canonical form
    selected = _normalize_method(args.method)

    # Map canonical to module name (update if you rename files)
    method_map = {
        "method1_yolo": "method1_yolo",
        "method2_topdown": "method2_topdown",
        "method3_posepipe": "method3_posepipe",
    }

    if selected not in method_map:
        raise SystemExit(f"❌ Unknown method '{args.method}'. "
                         f"Expected one of: {', '.join(sorted(method_map))}")

    # Validate n_overlay
    if args.n_overlay < 0:
        raise SystemExit("❌ n_overlay must be >= 0")

    # Ensure output directories exist
    _ensure_output_paths(args.out_json, args.vis_dir, args.out_mp4)

    # Import and dispatch
    target = method_map[selected]
    mod = _import_pipeline(target)
    return _dispatch(mod, args.in_mp4, args.out_json, args.vis_dir, args.n_overlay, args.out_mp4)


if __name__ == "__main__":
    sys.exit(main())
