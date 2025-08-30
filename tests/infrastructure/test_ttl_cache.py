from __future__ import annotations

import time

import pytest

from src.infrastructure.ttl_cache import TTLCache


def test_ttl_cache_miss_then_hit() -> None:
    cache: TTLCache[str, int] = TTLCache(ttl_seconds=1.0)
    assert cache.get("a") is None
    cache.set("a", 42)
    assert cache.get("a") == 42


def test_ttl_cache_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    cache: TTLCache[str, str] = TTLCache(ttl_seconds=0.5)
    cache.set("k", "v")

    real_mono = time.monotonic

    def later() -> float:
        return real_mono() + 1.0

    # Force expiry
    monkeypatch.setattr("time.monotonic", later)
    assert cache.get("k") is None
