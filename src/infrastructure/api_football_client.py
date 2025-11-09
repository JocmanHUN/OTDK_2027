from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Mapping, Optional, Protocol, cast

import requests

from src.config.settings import settings


class _HasHeaders(Protocol):
    headers: Mapping[str, str]


logger = logging.getLogger(__name__)
_TRUE_SET = {"1", "true", "yes", "on"}
try:
    _RATE_LIMIT_SLEEP_SECONDS = max(0.0, float(os.getenv("API_FOOTBALL_429_SLEEP_SECONDS", "60")))
except ValueError:
    _RATE_LIMIT_SLEEP_SECONDS = 60.0


def _env_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUE_SET


_GLOBAL_LOG_RESPONSES = _env_flag(os.getenv("API_FOOTBALL_LOG_RESPONSES"))


def set_api_response_logging(enabled: bool) -> None:
    """Globally enable/disable dumping raw API responses to stdout."""

    global _GLOBAL_LOG_RESPONSES
    _GLOBAL_LOG_RESPONSES = bool(enabled)


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
    - Optional stdout logging of every API response (for GUI debug sessions).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        *,
        log_responses: bool | None = None,
    ) -> None:
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.timeout = float(timeout if timeout is not None else settings.timeout)
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)
        self._log_responses = (
            _GLOBAL_LOG_RESPONSES if log_responses is None else bool(log_responses)
        )

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
        # Normalize params to strings for consistent API behavior
        norm_params = self._normalize_params(params)

        while attempt <= self.max_retries:
            try:
                resp = self._session.request(
                    method="GET", url=url, params=norm_params, timeout=self.timeout
                )
                self._log_http_response(url, norm_params, resp)

                # Success
                if 200 <= resp.status_code < 300:
                    if resp.headers.get("Content-Type", "").startswith("application/json"):
                        payload = resp.json()
                        if _payload_has_rate_limit_error(payload):
                            retry_after = self._compute_sleep_seconds(attempt, resp)
                            retry_after = max(
                                float(_RATE_LIMIT_SLEEP_SECONDS),
                                float(retry_after),
                            )
                            logger.warning(
                                "APIFootballClient GET %s hit JSON rateLimit error. Retrying in %.2fs "
                                "(attempt %d/%d)",
                                url,
                                retry_after,
                                attempt + 1,
                                self.max_retries,
                            )
                            attempt += 1
                            if attempt > self.max_retries:
                                break
                            time.sleep(retry_after)
                            continue
                        return payload
                    return resp.text

                # Rate limited or server error -> retry
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = self._compute_sleep_seconds(attempt, resp)
                    if resp.status_code == 429:
                        retry_after = max(float(_RATE_LIMIT_SLEEP_SECONDS), float(retry_after))
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

    def _normalize_params(self, params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if params is None:
            return None
        out: dict[str, Any] = {}
        for k, v in params.items():
            key = str(k)
            if isinstance(v, (int, float, bool)) or v is None:
                out[key] = str(v) if v is not None else v
            else:
                out[key] = v
        return out

    def _log_http_response(
        self,
        url: str,
        params: Optional[Mapping[str, Any]],
        resp: requests.Response,
    ) -> None:
        if not self._log_responses:
            return
        try:
            ts = datetime.now().isoformat(timespec="seconds")
        except Exception:
            ts = ""
        prefix = f"[API RESPONSE] {ts} GET {url} status={resp.status_code}"
        if params:
            prefix = f"{prefix} params={dict(params)}"
        print(prefix, flush=True)
        try:
            print(resp.text, flush=True)
        except Exception as exc:
            print(f"<unable to read body: {exc}>", flush=True)


def _payload_has_rate_limit_error(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    errors = payload.get("errors")
    if not isinstance(errors, Mapping):
        return False
    for key, value in errors.items():
        key_str = str(key).lower()
        if key_str in {"ratelimit", "rate_limit"} and value:
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, Mapping):
                return bool(value)
            if isinstance(value, list):
                return bool(value)
            return True
    return False
