import logging
def get_logger(name="gait", level="INFO"):
    log=logging.getLogger(name); 
    if not log.handlers:
        h=logging.StreamHandler(); h.setFormatter(logging.Formatter("%(levelname)s %(message)s")); log.addHandler(h)
    log.setLevel(getattr(logging, level, logging.INFO))
    return log
