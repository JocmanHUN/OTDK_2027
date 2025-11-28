from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _iso_date_for_tomorrow() -> str:
    # Europe/Budapest local tomorrow simplified: use UTC + 1 day
    return (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()


def ensure_schema(conn: sqlite3.Connection) -> None:
    # Touch all repos so their CREATE TABLE IF NOT EXISTS run
    from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
    from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite
    from src.repositories.sqlite.matches_sqlite import MatchesRepoSqlite
    from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite
    from src.repositories.sqlite.predictions_sqlite import PredictionsRepoSqlite

    LeaguesRepoSqlite(conn)
    BookmakersRepoSqlite(conn)
    MatchesRepoSqlite(conn)
    OddsRepoSqlite(conn)
    PredictionsRepoSqlite(conn)


def _load_whitelisted_leagues(conn: sqlite3.Connection) -> set[int]:
    cur = conn.execute("SELECT league_id FROM leagues")
    return {int(r[0]) for r in cur.fetchall()}


def _filter_fixtures_by_leagues(
    fixtures: Iterable[Mapping[str, Any]], league_ids: set[int]
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for f in fixtures:
        lid_val = f.get("league_id")
        if not isinstance(lid_val, (int, str)):
            continue
        lid_i = int(lid_val)
        if lid_i in league_ids:
            out.append(f)
    return out


def _upsert_match(conn: sqlite3.Connection, f: Mapping[str, Any]) -> None:
    from src.repositories.matches import Match as DBMatch
    from src.repositories.sqlite.matches_sqlite import MatchesRepoSqlite

    repo = MatchesRepoSqlite(conn)
    fx_raw = f.get("fixture_id")
    if not isinstance(fx_raw, (int, str)):
        return
    fx_id = int(fx_raw)
    existing = repo.get_by_id(fx_id)
    dt = f.get("date_utc")
    if not isinstance(dt, datetime):
        dt = datetime.now(timezone.utc)
    if existing is None:
        lg_val = f.get("league_id")
        if isinstance(lg_val, (int, str)):
            league_id = int(lg_val)
        else:
            league_id = 0
        ss_val = f.get("season") if f.get("season") is not None else f.get("season_year")
        season = int(ss_val) if isinstance(ss_val, (int, str)) else 0
        home_team = str(f.get("home_name") or f.get("home_id") or "")
        away_team = str(f.get("away_name") or f.get("away_id") or "")
        repo.insert(
            DBMatch(
                id=fx_id,
                league_id=league_id,
                season=season,
                date=dt,
                home_team=home_team,
                away_team=away_team,
                real_result=None,
            )
        )
    else:
        # no update method for details; skip to keep it minimal
        pass


def _upsert_bookmakers(conn: sqlite3.Connection, bm: dict[int, str]) -> None:
    from src.repositories.bookmakers import Bookmaker
    from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite

    repo = BookmakersRepoSqlite(conn)
    for bid, name in bm.items():
        existing = repo.get_by_id(int(bid))
        if existing is None:
            repo.insert(Bookmaker(id=int(bid), name=str(name)))
        elif existing.name != name:
            repo.update(Bookmaker(id=int(bid), name=str(name)))


def _save_odds(conn: sqlite3.Connection, match_id: int, odds_list: list) -> None:
    from src.repositories.odds import Odds as DBOdds
    from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite

    repo = OddsRepoSqlite(conn)
    # Build existing map bookmaker_id -> odds_id
    existing = {o.bookmaker_id: o for o in repo.list_by_match(match_id)}
    for o in odds_list:
        # o is domain Odds (Decimal); convert
        try:
            bid = int(o.bookmaker_id)
            home = float(o.home)
            draw = float(o.draw)
            away = float(o.away)
        except Exception:
            continue
        if bid in existing:
            db_obj = existing[bid]
            db_obj.match_id = match_id
            db_obj.bookmaker_id = bid
            db_obj.home = home
            db_obj.draw = draw
            db_obj.away = away
            repo.update(db_obj)
        else:
            repo.insert(
                DBOdds(
                    id=None, match_id=match_id, bookmaker_id=bid, home=home, draw=draw, away=away
                )
            )


def _compute_and_store_predictions(
    conn: sqlite3.Connection,
    f: Mapping[str, Any],
    odds_list: list[Any] | list,
    *,
    log_date: str | None = None,
) -> None:
    from src.application.services.export_logging import log_features_csv
    from src.application.services.history_service import HistoryService
    from src.application.services.prediction_pipeline import (
        ContextBuilder,
        PredictionAggregatorImpl,
    )
    from src.domain.entities.match import Match
    from src.domain.value_objects.enums import MatchStatus
    from src.domain.value_objects.ids import FixtureId, LeagueId
    from src.models import default_models
    from src.repositories.predictions import Prediction as DBPred
    from src.repositories.sqlite.predictions_sqlite import PredictionsRepoSqlite

    if not odds_list:
        return

    fx_raw = f.get("fixture_id")
    lg_raw = f.get("league_id")
    ss_raw = f.get("season") if f.get("season") is not None else f.get("season_year")
    if not isinstance(fx_raw, (int, str)) or not isinstance(lg_raw, (int, str)):
        return
    fx_id = int(fx_raw)
    league_id = int(lg_raw)
    season = int(ss_raw) if isinstance(ss_raw, (int, str)) else 0
    home_id_val = f.get("home_id")
    away_id_val = f.get("away_id")
    if not isinstance(home_id_val, (int, str)) or not isinstance(away_id_val, (int, str)):
        return

    home_id = int(home_id_val)
    away_id = int(away_id_val)

    dt = f.get("date_utc")
    if not isinstance(dt, datetime):
        dt = datetime.now(timezone.utc)

    # Build context
    history = HistoryService()

    # 0) Threshold checks before any prediction:
    # - at least 10 recent (finished, league) matches for both teams
    # - at least 5 head-to-head (excluding friendlies)
    home_i = int(home_id)
    away_i = int(away_id)
    home_rows = history.get_recent_team_stats(home_i, league_id, season, 10)
    away_rows = history.get_recent_team_stats(away_i, league_id, season, 10)
    if len(home_rows) < 10 or len(away_rows) < 10:
        return  # skip predicting due to insufficient team history
    h2h_rows = history.get_head_to_head(home_i, away_i, last=50, exclude_friendlies=True)
    if len(h2h_rows) < 5:
        return  # skip predicting due to insufficient H2H

    ctx_builder = ContextBuilder(history=history)
    ctx = ctx_builder.build_from_meta(
        fixture_id=fx_id,
        league_id=league_id,
        season=season,
        home_team_id=home_i,
        away_team_id=away_i,
    )

    if log_date:
        try:
            log_features_csv(log_date, f, ctx)
        except Exception:
            pass

    # Minimal domain match for models
    match = Match(
        fixture_id=FixtureId(fx_id),
        league_id=LeagueId(league_id),
        season=season,
        kickoff_utc=dt,
        home_name=str(f.get("home_name") or home_i or ""),
        away_name=str(f.get("away_name") or away_i or ""),
        status=MatchStatus.SCHEDULED,
    )

    models = default_models()
    agg = PredictionAggregatorImpl()
    preds = agg.run_all(models, match, ctx)

    # Store predictions with status OK only
    repo = PredictionsRepoSqlite(conn)
    for p in preds:
        probs = getattr(p, "probs", None)
        if probs is None:
            continue
        # pick predicted result by max prob
        pr = probs
        res = "1"
        if pr.draw >= pr.home and pr.draw >= pr.away:
            res = "X"
        elif pr.away >= pr.home and pr.away >= pr.draw:
            res = "2"
        dbp = DBPred(
            id=None,
            match_id=fx_id,
            model_name=str(getattr(p.model, "value", p.model)),
            prob_home=float(pr.home),
            prob_draw=float(pr.draw),
            prob_away=float(pr.away),
            predicted_result=res,
            is_correct=None,
        )
        repo.insert(dbp)


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()

    parser = argparse.ArgumentParser(description="Daily pipeline: fixtures → odds → predictions")
    parser.add_argument("--db", default=os.path.join("data", "odtk.sqlite3"))
    parser.add_argument(
        "--date", default=_iso_date_for_tomorrow(), help="YYYY-MM-DD (Europe/Budapest)"
    )
    parser.add_argument(
        "--with-predictions", action="store_true", help="Compute and store predictions"
    )
    args = parser.parse_args(argv)

    db_path = os.path.abspath(args.db)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    from src.application.services.export_logging import log_odds_csv
    from src.application.services.fixtures_service import FixturesService
    from src.application.services.odds_service import OddsService

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)

        whitelisted = _load_whitelisted_leagues(conn)

        fx_svc = FixturesService()
        raw = fx_svc.get_daily_fixtures(args.date)

        fixtures = _filter_fixtures_by_leagues(raw, whitelisted)
        if not fixtures:
            print("No fixtures after league filtering.")
            return 0

        odds_svc = OddsService()

        for f in fixtures:
            fx_val = f.get("fixture_id")
            if not isinstance(fx_val, (int, str)):
                continue
            fx_id_i = int(fx_val)

            # First, fetch odds. Only persist anything if at least one 1X2 odds exists.
            odds_list = odds_svc.get_fixture_odds(fx_id_i)
            if not odds_list:
                continue

            try:
                log_odds_csv(args.date, f, odds_list)
            except Exception:
                pass

            # Persist match only once we know there are odds
            _upsert_match(conn, f)

            # Upsert bookmakers from same payload
            bm = odds_svc.get_fixture_bookmakers(fx_id_i)
            if bm:
                _upsert_bookmakers(conn, bm)

            # Save odds per bookmaker (update if exists)
            _save_odds(conn, fx_id_i, odds_list)

            if args.with_predictions:
                _compute_and_store_predictions(conn, f, odds_list, log_date=args.date)

    finally:
        conn.close()

    print("Daily pipeline done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
