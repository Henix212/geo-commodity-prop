"""Shared logging: console + rotating file under data/."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from backend import config


_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotent logging setup used by main and CLIs."""
    global _CONFIGURED
    if _CONFIGURED:
        root = logging.getLogger()
        if level:
            root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    config.ensure_dirs()
    lvl = getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = config.DATA_DIR / "app.log"
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True
