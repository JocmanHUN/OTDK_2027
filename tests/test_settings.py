import importlib
import sys
import pytest

MODULE = "src.config.settings"


def reload_settings():
    return importlib.reload(importlib.import_module(MODULE))


def test_direct_api_settings(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.delenv("USE_RAPIDAPI", raising=False)
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    module = reload_settings()
    assert module.settings.base_url == module.DIRECT_BASE_URL
    assert module.settings.headers == {"x-apisports-key": "secret"}


def test_rapidapi_settings(monkeypatch):
    monkeypatch.setenv("USE_RAPIDAPI", "true")
    monkeypatch.setenv("RAPIDAPI_KEY", "rk")
    monkeypatch.delenv("API_KEY", raising=False)
    module = reload_settings()
    assert module.settings.base_url == module.RAPIDAPI_BASE_URL
    assert module.settings.headers == {
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        "x-rapidapi-key": "rk",
    }


def test_rapidapi_requires_key(monkeypatch):
    monkeypatch.setenv("USE_RAPIDAPI", "true")
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        reload_settings()


def test_direct_requires_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "temp")
    module = reload_settings()
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("RAPIDAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API access key provided"):
        module._build_settings()
    monkeypatch.setenv("RAPIDAPI_KEY", "rk")
    with pytest.raises(RuntimeError, match="API_KEY is required"):
        module._build_settings()
