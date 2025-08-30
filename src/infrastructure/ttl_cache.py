from __future__ import annotations

import time
from typing import Dict, Generic, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Simple in-memory TTL cache.

    - Stores values with an absolute expiry computed from ``ttl_seconds``.
    - Uses ``time.monotonic()`` for steady time measurement.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = float(ttl_seconds)
        self._store: Dict[K, Tuple[float, V]] = {}

    def get(self, key: K) -> Optional[V]:
        now = time.monotonic()
        item = self._store.get(key)
        if item is None:
            return None
        expiry, value = item
        if now >= expiry:
            # Expired
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: K, value: V) -> None:
        expiry = time.monotonic() + self._ttl
        self._store[key] = (expiry, value)
