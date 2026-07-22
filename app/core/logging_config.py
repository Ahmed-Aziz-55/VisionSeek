"""centralized logging setup for VisionSeek. Call `setup_logging()` once, at
the entry point of any script — before running pipeline/service code —
so every module's `logging.getLogger(__name__)` calls share the same
format and handlers.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | None = "logs/visionseek.log") -> None:
    """
    Configures the root logger with a console handler (always) and an
    optional file handler. Individual modules never call this themselves —
    they only do `logger = logging.getLogger(__name__)` and log through it.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any prior basicConfig set by an imported library
    )