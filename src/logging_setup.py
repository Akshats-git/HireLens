"""
Loguru configuration for HireLens entry points.

Only entry points (CLI scripts, the FastAPI app) should call `configure_logging`.
Library modules must not touch handlers: `logger.remove()` is global, so a module
that reconfigures logging at import time silently discards the handlers its
importer installed.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import PROJECT_ROOT, load_config

LOG_DIR = PROJECT_ROOT / "logs"


def configure_logging(log_file: str | None = None, level: str | None = None) -> None:
    """
    Install stderr (and optionally file) log handlers, replacing any existing ones.

    Args:
        log_file: Filename under logs/ for a DEBUG-level rotating log. Console
                  only if omitted.
        level: Console log level. Falls back to the configured default.
    """
    settings = load_config().get("logging", {})
    console_level = level or settings.get("level", "INFO")
    log_format = settings.get(
        "format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}",
    )

    logger.remove()
    logger.add(sys.stderr, level=console_level, format=log_format)

    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.add(
            Path(LOG_DIR) / log_file,
            level="DEBUG",
            format=log_format,
            rotation=settings.get("rotation", "100 MB"),
            retention=settings.get("retention", "30 days"),
        )
