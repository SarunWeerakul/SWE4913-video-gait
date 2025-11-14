# src/common/io.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

# --- core helpers -------------------------------------------------------------

def ensure_dir(p: str | os.PathLike | None) -> None:
    """Create directory p if needed (no-op for '', None)."""
    if not p:
        return
    Path(p).mkdir(parents=True, exist_ok=True)

def read_json(p: str | os.PathLike, default: Any = None) -> Any:
    """
    Read UTF-8 JSON. If file doesn't exist and default is provided, return default.
    """
    path = Path(p)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(p: str | os.PathLike, obj: Any, pretty: bool = True, atomic: bool = True) -> None:
    """
    Write JSON (UTF-8). Creates parent dirs. Atomic write avoids partial files.
    """
    path = Path(p)
    ensure_dir(path.parent)
    text = json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)  # atomic on POSIX

# --- convenient extras (optional but useful) ----------------------------------

def read_text(p: str | os.PathLike, default: Optional[str] = None) -> Optional[str]:
    path = Path(p)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")

def write_text(p: str | os.PathLike, s: str) -> None:
    path = Path(p)
    ensure_dir(path.parent)
    path.write_text(s, encoding="utf-8")

def list_videos(folder: str | os.PathLike, pattern: str = "camera_*.mp4") -> list[Path]:
    """Sorted list of video files (Path objects)."""
    return sorted(Path(folder).glob(pattern))
