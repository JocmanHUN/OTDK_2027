import json
import logging
from collections.abc import Generator
from logging.handlers import RotatingFileHandler

import pytest
from _pytest.logging import LogCaptureFixture

from src.logging_config import LOG_NAME, CacheStats, JsonFormatter, get_logger


@pytest.fixture(autouse=True)
def reset_logger_handlers() -> Generator[None, None, None]:
    """Ensure tests run with a clean logger state."""
    logger = logging.getLogger(LOG_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    yield
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def test_json_formatter_returns_json_with_extras() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc"
    record.user = "alice"
    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["level"] == "INFO"
    assert data["logger"] == "test"
    assert data["message"] == "hello"
    assert data["request_id"] == "abc"
    assert data["extra"]["user"] == "alice"


def test_get_logger_configures_two_handlers() -> None:
    logger = get_logger()
    assert len(logger.handlers) == 2
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def test_cache_stats_hit_miss_logic() -> None:
    stats = CacheStats()
    assert stats.hit_rate == 0.0
    stats.record_hit()
    assert stats.hit_rate == 100.0
    stats.record_miss()
    assert stats.hit_rate == 50.0


def test_cache_stats_logging(caplog: LogCaptureFixture) -> None:
    stats = CacheStats()
    stats.record_hit()
    stats.record_miss()
    expected_rate = round(stats.hit_rate, 2)
    logger = get_logger()
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=LOG_NAME)
    stats.log_hit_rate()
    logger.removeHandler(caplog.handler)
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "Cache hit-rate"
    record.hit_rate = expected_rate
    formatter = JsonFormatter()
    data = json.loads(formatter.format(record))
    assert data["extra"]["hit_rate"] == expected_rate
