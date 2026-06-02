"""Утилиты логирования для CLI и пакетных сценариев."""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Создает или возвращает настроенный логгер.

    По умолчанию логгер пишет в консоль и при необходимости может писать
    в файл журнала. Повторные вызовы с тем же именем логгера переиспользуют
    существующие обработчики, чтобы избежать дублирования строк журнала.

    Args:
        name: Имя логгера, обычно ``__name__`` или метка приложения.
        log_file: Необязательный путь к файлу журнала. Родительские каталоги
            создаются автоматически при необходимости.
        level: Уровень логирования, применяемый к логгеру и его обработчикам.

    Returns:
        Настроенный экземпляр ``logging.Logger``.
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
