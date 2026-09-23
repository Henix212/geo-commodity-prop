"""Console + rotating file logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from backend import config


def setup_logging(level: str | None = None, *, name: str | None = None) -> logging.Logger:
    config.ensure_dirs()
    level_name = (level or config.LOG_LEVEL).upper()
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(getattr(logging, level_name, logging.INFO))
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
        file_handler = RotatingFileHandler(
            config.LOG_DIR / "geo_commodity.log",
            maxBytes=5_000_000,
            backupCount=3,
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    else:
        root.setLevel(getattr(logging, level_name, logging.INFO))
    return logging.getLogger(name) if name else root
