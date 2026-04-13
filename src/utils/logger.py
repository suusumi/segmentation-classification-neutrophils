"""Logging helpers for CLI and batch workflows."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    The logger writes to the console by default and can optionally write to
    a log file. Repeated calls with the same logger name reuse the existing
    handlers to avoid duplicate log lines.

    Args:
        name: Logger name, typically ``__name__`` or an application label.
        log_file: Optional path to a log file. Parent directories are created
            automatically if needed.
        level: Logging level applied to the logger and its handlers.

    Returns:
        A configured ``logging.Logger`` instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if getattr(logger, "_blood_project_configured", False):
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    logger._blood_project_configured = True  # type: ignore[attr-defined]
    return logger
