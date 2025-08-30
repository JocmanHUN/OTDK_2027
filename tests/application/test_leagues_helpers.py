from __future__ import annotations

from typing import Any, Mapping, cast

from src.application.services.leagues_service import (
    _has_odds_and_stats,
    _parse_leagues_response,
    _safe_int,
)


def test_parse_leagues_response_ok() -> None:
    payload: dict[str, Any] = {"response": [{"league": {"id": 1}, "seasons": []}]}
    out = _parse_leagues_response(payload)
    assert isinstance(out, list) and len(out) == 1


def test_parse_leagues_response_bad_shape() -> None:
    bad: Any
    for bad in (None, {}, {"response": {}}, {"wrong": []}):
        out = _parse_leagues_response(cast(Any, bad))
        assert out == []


def test_has_odds_and_stats_variants() -> None:
    # Both True
    cov: Mapping[str, Any] = {"odds": True, "fixtures": {"statistics": True}}
    assert _has_odds_and_stats(cov) is True

    # Accept statistics_fixtures variant
    cov2: Mapping[str, Any] = {"odds": True, "fixtures": {"statistics_fixtures": True}}
    assert _has_odds_and_stats(cov2) is True

    # Missing odds
    cov3: Mapping[str, Any] = {"fixtures": {"statistics": True}}
    assert _has_odds_and_stats(cov3) is False


def test_safe_int() -> None:
    assert _safe_int(5) == 5
    assert _safe_int("7") == 7
    assert _safe_int(None, default=99) == 99
