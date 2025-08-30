from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest


def _reload_settings() -> Any:
    # Remove cached module to force re-evaluation of settings on import
    if "src.config.settings" in sys.modules:
        del sys.modules["src.config.settings"]
    import src.config.settings as settings_module

    importlib.reload(settings_module)
    return settings_module


def test_direct_api_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prevent picking up values from a real .env during the test
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("API_KEY", "k123")
    monkeypatch.delenv("USE_RAPIDAPI", raising=False)
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)

    settings_module = _reload_settings()
    s = settings_module.settings

    assert s.base_url == settings_module.DIRECT_BASE_URL
    assert s.headers.get("x-apisports-key") == "k123"
    assert s.timeout == settings_module.REQUEST_TIMEOUT


def test_missing_all_keys_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    monkeypatch.setenv("USE_RAPIDAPI", "false")

    with pytest.raises(RuntimeError):
        _ = _reload_settings()


def test_rapidapi_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("USE_RAPIDAPI", "true")
    monkeypatch.setenv("RAPIDAPI_KEY", "rk123")
    monkeypatch.setenv("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
    monkeypatch.delenv("API_KEY", raising=False)

    settings_module = _reload_settings()
    s = settings_module.settings
    assert s.base_url == settings_module.RAPIDAPI_BASE_URL
    assert s.headers.get("x-rapidapi-key") == "rk123"
    assert s.headers.get("x-rapidapi-host") == "api-football-v1.p.rapidapi.com"


def test_rapidapi_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    monkeypatch.setenv("USE_RAPIDAPI", "true")
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        _ = _reload_settings()
