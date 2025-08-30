from __future__ import annotations

import logging
import random
import time
from typing import Any, Mapping, Optional, Protocol, cast

import requests

from src.config.settings import settings


class _HasHeaders(Protocol):
    headers: Mapping[str, str]


logger = logging.getLogger(__name__)


class APIError(RuntimeError):
    """Raised when the API-FOOTBALL API returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIFootballClient:
    """Minimal API-FOOTBALL client with auth, timeout, and retries.

    Features:
    - Auth via ``x-apisports-key`` or RapidAPI headers (from settings).
    - Configurable timeout, retry count and exponential backoff with jitter.
    - Special handling for HTTP 429 and 5xx responses.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.timeout = float(timeout if timeout is not None else settings.timeout)
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)

        self._session = requests.Session()
        # Merge default headers from settings with any provided overrides
        default_headers = dict(settings.headers)
        if headers:
            default_headers.update(headers)
        self._session.headers.update(default_headers)

    def _full_url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def get(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        """Perform a GET request and return decoded JSON.

        Retries on 429 and 5xx responses with exponential backoff.
        Raises APIError on persistent failures or non-retriable 4xx.
        """

        url = self._full_url(path)
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt <= self.max_retries:
            try:
                resp = self._session.request(
                    method="GET", url=url, params=params, timeout=self.timeout
                )

                # Success
                if 200 <= resp.status_code < 300:
                    if resp.headers.get("Content-Type", "").startswith("application/json"):
                        return resp.json()
                    return resp.text

                # Rate limited or server error -> retry
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = self._compute_sleep_seconds(attempt, resp)
                    logger.warning(
                        "APIFootballClient GET %s failed with %s. Retrying in %.2fs (attempt %d/%d)",
                        url,
                        resp.status_code,
                        retry_after,
                        attempt + 1,
                        self.max_retries,
                    )
                    attempt += 1
                    if attempt > self.max_retries:
                        break
                    time.sleep(retry_after)
                    continue

                # Non-retriable client error
                raise APIError(
                    f"API error {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                retry_after = self._compute_sleep_seconds(attempt)
                logger.warning(
                    "APIFootballClient GET %s exception: %s. Retrying in %.2fs (attempt %d/%d)",
                    url,
                    type(exc).__name__,
                    retry_after,
                    attempt + 1,
                    self.max_retries,
                )
                attempt += 1
                if attempt > self.max_retries:
                    break
                time.sleep(retry_after)

        # Exceeded retries
        if last_error is not None:
            raise APIError(f"Request failed after retries: {last_error}")
        raise APIError("Request failed after retries", status_code=None)

    def get_status(self) -> Any:
        """GET /status endpoint convenience method."""
        return self.get("status")

    def _compute_sleep_seconds(self, attempt: int, response: Optional[_HasHeaders] = None) -> float:
        """Compute sleep duration for retries.

        - Respect Retry-After header if provided and valid.
        - Otherwise exponential backoff: backoff_factor * (2**attempt) + jitter.
        """
        if response is not None:
            # Cast headers to a precise Mapping to avoid "Any" propagation in type checking
            headers = cast(Mapping[str, str], response.headers)
            ra = headers.get("Retry-After")
            if ra:
                try:
                    # Retry-After in seconds (integer)
                    return max(0.0, float(int(ra)))
                except (TypeError, ValueError):
                    pass

        # Exponential backoff with small jitter
        base: float = float(self.backoff_factor) * float(2**attempt)
        jitter: float = float(random.uniform(0.0, 0.1))
        return float(base + jitter)
