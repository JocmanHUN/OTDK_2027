"""Logging configuration for the OTDK project.

Provides a JSON formatted logger named ``otdk`` and simple cache statistics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_NAME = "otdk"
LOG_FILE = Path("logs/app.log")
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

# Attributes present on every LogRecord. Anything else is considered an extra field.
DEFAULT_LOG_RECORD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Formatter returning log records as JSON strings."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - short description
        base: dict[str, Any] = {
            "time": datetime.utcnow().isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in DEFAULT_LOG_RECORD_ATTRS
        }
        request_id = extras.pop("request_id", None)
        if request_id is not None:
            base["request_id"] = request_id
        if extras:
            base["extra"] = extras
        return json.dumps(base, ensure_ascii=False, default=str)


def get_logger() -> logging.Logger:
    """Return a configured logger for the project."""
    logger = logging.getLogger(LOG_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class CacheStats:
    """Simple cache hit/miss statistics collector."""

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._logger = get_logger()

    def record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a percentage."""
        total = self._hits + self._misses
        return (self._hits / total * 100) if total else 0.0

    def log_hit_rate(self) -> None:
        """Log the current cache hit rate."""
        self._logger.info("Cache hit-rate", extra={"hit_rate": round(self.hit_rate, 2)})
