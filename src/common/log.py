# src/common/log.py
import logging
from typing import Optional

_DEFAULT_FMT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"

def get_logger(
    name: str = "gait",
    level: str | int = "INFO",
    to_file: Optional[str] = None,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> logging.Logger:
    """
    Create/reuse a logger with a single stream handler and optional file handler.
    Usage:
        log = get_logger("pose.yolo", "DEBUG")
        log.info("hello")
    """
    log = logging.getLogger(name)

    # Normalize level
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    log.setLevel(lvl)

    # Add handlers only once
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        sh.setLevel(lvl)
        log.addHandler(sh)

    if to_file and not any(isinstance(h, logging.FileHandler) and getattr(h, "_path", None) == to_file
                           for h in log.handlers):
        fh = logging.FileHandler(to_file)
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        fh.setLevel(lvl)
        fh._path = to_file  # mark so we don’t add duplicates
        log.addHandler(fh)

    return log
