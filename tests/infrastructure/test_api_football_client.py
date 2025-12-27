# mypy: ignore-errors

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import requests

import src.infrastructure.api_football_client as api_mod

# Ensure direct API key path is configured before importing the client
os.environ.setdefault("API_KEY", "dummy")
os.environ.pop("USE_RAPIDAPI", None)
os.environ.pop("RAPIDAPI_KEY", None)

from src.infrastructure.api_football_client import APIError, APIFootballClient


class _FakeResponse:
    def __init__(
        self,
        status_code: int,
        json_data: Any | None = None,
        text: str = "",
        headers: Dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self.headers = headers or {"Content-Type": "application/json"}

    def json(self) -> Any:
        return self._json


def test_get_status_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        calls.append({"method": method, "url": url, "params": params, "timeout": timeout})
        return _FakeResponse(200, json_data={"response": "ok"})

    # Patch requests.Session.request on the instance level
    client = APIFootballClient()
    monkeypatch.setattr(client._session, "request", fake_request)

    data = client.get_status()
    assert data == {"response": "ok"}
    assert calls and calls[0]["url"].endswith("/status")
    assert calls[0]["method"] == "GET"


def test_retry_on_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(429, headers={"Retry-After": "0", "Content-Type": "application/json"}),
        _FakeResponse(200, json_data={"ok": True}),
    ]

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    # Avoid waiting during tests
    monkeypatch.setattr("time.sleep", lambda s: None)

    data = client.get("status")
    assert data == {"ok": True}


def test_retry_on_5xx_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(500, text="server error", headers={"Content-Type": "text/plain"}),
        _FakeResponse(200, json_data={"ok": 1}),
    ]

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", lambda s: None)

    data = client.get("status")
    assert data == {"ok": 1}


def test_non_retriable_4xx_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return _FakeResponse(400, text="bad request", headers={"Content-Type": "text/plain"})

    client = APIFootballClient()
    monkeypatch.setattr(client._session, "request", fake_request)

    with pytest.raises(APIError) as exc:
        client.get("status")

    assert "400" in str(exc.value)


def test_text_response_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return _FakeResponse(200, text="OK", headers={"Content-Type": "text/plain"})

    client = APIFootballClient()
    monkeypatch.setattr(client._session, "request", fake_request)

    data = client.get("status")
    assert data == "OK"


def test_retry_exhaustion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(500, text="e1", headers={"Content-Type": "text/plain"}),
        _FakeResponse(500, text="e2", headers={"Content-Type": "text/plain"}),
        _FakeResponse(500, text="e3", headers={"Content-Type": "text/plain"}),
    ]

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(APIError):
        client.get("status")


def test_429_without_retry_after_waits_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(429, text="rate", headers={"Content-Type": "text/plain"}),
        _FakeResponse(200, json_data={"ok": True}),
    ]

    sleep_calls: list[float] = []

    def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", fake_sleep)

    data = client.get("status")
    assert data == {"ok": True}
    assert sleep_calls and abs(sleep_calls[0] - 60.0) < 1e-6


def test_429_retry_after_longer_than_minimum_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(
            429, text="rate", headers={"Retry-After": "120", "Content-Type": "text/plain"}
        ),
        _FakeResponse(200, json_data={"ok": True}),
    ]

    sleep_calls: list[float] = []

    def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", fake_sleep)

    data = client.get("status")
    assert data == {"ok": True}
    assert sleep_calls and abs(sleep_calls[0] - 120.0) < 1e-6


def test_json_rate_limit_error_triggers_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(
            200,
            json_data={
                "errors": {
                    "rateLimit": "Too many requests. Your rate limit is 300 requests per minute."
                }
            },
        ),
        _FakeResponse(200, json_data={"ok": True}),
    ]

    sleep_calls: list[float] = []

    def fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", fake_sleep)

    data = client.get("fixtures")
    assert data == {"ok": True}
    assert sleep_calls and abs(sleep_calls[0] - 60.0) < 1e-6


def test_backoff_without_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(500, text="err", headers={"Content-Type": "text/plain"}),
        _FakeResponse(200, json_data={"ok": 1}),
    ]

    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    def fake_request(
        method: str, url: str, params: Any | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        return seq.pop(0)

    client = APIFootballClient(max_retries=2, backoff_factor=0.75)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

    data = client.get("status")
    assert data == {"ok": 1}
    # Attempt 0 backoff = backoff_factor * (2**0) = 0.75
    assert sleeps and abs(sleeps[0] - 0.75) < 1e-6


def test_env_flag_and_bad_sleep_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_FOOTBALL_429_SLEEP_SECONDS", "not-a-number")
    spec = importlib.util.spec_from_file_location(
        "api_client_temp", Path("src/infrastructure/api_football_client.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._RATE_LIMIT_SLEEP_SECONDS == 60.0
    assert module._env_flag("yes") is True
    assert module._env_flag(None) is False


def test_throttle_per_second_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    client = APIFootballClient()
    monkeypatch.setattr(api_mod, "_MAX_CALLS_PER_SEC", 1)
    sleep_calls: list[float] = []
    monkeypatch.setattr(api_mod.time, "sleep", lambda s: sleep_calls.append(s))
    # Force a sleep of 0.1s
    monkeypatch.setattr(api_mod.time, "monotonic", lambda: 100.0)
    client._request_times_sec.clear()
    client._request_times_sec.append(99.1)
    client._throttle_per_second()
    assert sleep_calls and pytest.approx(sleep_calls[0], rel=0.0, abs=1e-6) == 0.1
    # Disable throttle entirely
    monkeypatch.setattr(api_mod, "_MAX_CALLS_PER_SEC", 0)
    client._throttle_per_second()
    assert len(sleep_calls) == 1


def test_headers_override_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    client = APIFootballClient(headers={"X-Test": "1"})
    assert client._session.headers.get("X-Test") == "1"


def test_daily_rate_limit_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResponse(
        200, json_data={"errors": {"request": "You have reached your requests per day quota."}}
    )
    client = APIFootballClient(max_retries=1, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", lambda *args, **kwargs: resp)
    with pytest.raises(APIError) as exc:
        client.get("status")
    assert exc.value.status_code == 429


def test_rate_limit_breaks_after_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResponse(200, json_data={"errors": {"ratelimit": "Too many requests per minute"}})
    client = APIFootballClient(max_retries=0, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", lambda *args, **kwargs: resp)
    monkeypatch.setattr(api_mod.time, "sleep", lambda s: None)
    with pytest.raises(APIError):
        client.get("fixtures")


def test_timeouts_raise_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    def fake_request(*args: Any, **kwargs: Any) -> _FakeResponse:
        raise requests.Timeout("boom")

    client = APIFootballClient(max_retries=1, backoff_factor=0)
    monkeypatch.setattr(client._session, "request", fake_request)
    monkeypatch.setattr(api_mod.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(APIError) as exc:
        client.get("status")

    assert "boom" in str(exc.value)
    assert sleeps  # slept once for retry


def test_compute_sleep_invalid_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
    client = APIFootballClient(backoff_factor=1.0)
    resp = _FakeResponse(429, headers={"Retry-After": "abc"})
    assert client._compute_sleep_seconds(1, resp) == pytest.approx(2.0)


def test_normalize_params_and_logging(capsys: pytest.CaptureFixture[str]) -> None:
    client = APIFootballClient(log_responses=True)
    params = client._normalize_params({"a": 1, "b": False, "c": None, "d": "x"})
    assert params == {"a": "1", "b": "False", "c": None, "d": "x"}

    class _Resp:
        status_code = 200
        headers: Dict[str, str] = {}

        @property
        def text(self) -> str:
            raise RuntimeError("boom body")

    client._log_http_response("http://t", params, _Resp())  # should print prefix and error
    out = capsys.readouterr().out
    assert "API RESPONSE" in out and "unable to read body" in out


def test_extract_rate_limit_level_variants() -> None:
    assert (
        api_mod._extract_rate_limit_level({"errors": {"requests": ["per minute max"]}}) == "minute"
    )
    payload = {"errors": {"rate_limit": {"a": "per day max"}}}
    assert api_mod._extract_rate_limit_level(payload) == "daily"
