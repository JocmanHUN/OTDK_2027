import os
from typing import Any, Dict, List

import pytest

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


def test_retry_after_header_used(monkeypatch: pytest.MonkeyPatch) -> None:
    seq = [
        _FakeResponse(429, text="rate", headers={"Retry-After": "2", "Content-Type": "text/plain"}),
        _FakeResponse(200, json_data={"ok": True}),
    ]

    sleep_calls: list[float] = []

    def fake_sleep(s: float) -> None:  # noqa: D401
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
    assert sleep_calls and abs(sleep_calls[0] - 2.0) < 1e-6


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
