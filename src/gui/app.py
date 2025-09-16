﻿from __future__ import annotations

import os
import sqlite3
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Mapping, Tuple, cast

from src.application.services.fixtures_service import FixturesService
from src.application.services.history_service import HistoryService
from src.application.services.leagues_service import LeaguesService
from src.application.services.odds_service import OddsService
from src.application.services.prediction_pipeline import ContextBuilder, PredictionAggregatorImpl
from src.domain.entities.match import Match as DomainMatch
from src.domain.value_objects.enums import MatchStatus
from src.domain.value_objects.ids import FixtureId, LeagueId
from src.models import default_models
from src.repositories.bookmakers import Bookmaker
from src.repositories.leagues import League
from src.repositories.matches import Match as DbMatch
from src.repositories.odds import Odds as RepoOdds
from src.repositories.predictions import Prediction as DbPrediction
from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
from src.repositories.sqlite.leagues_sqlite import LeaguesRepoSqlite
from src.repositories.sqlite.matches_sqlite import MatchesRepoSqlite
from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite
from src.repositories.sqlite.predictions_sqlite import PredictionsRepoSqlite

# Use zoneinfo lazily at call sites to avoid mypy issues with optional imports.


BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = str(BASE_DIR / "data" / "app.db")
MIGRATION_FILE = str(BASE_DIR / "migrations" / "V1__base.sql")


# ---------------------------- UI helpers ----------------------------
_MOJIBAKE_MAP: Dict[str, str] = {
    "Holnapi meccsek ?s predikci?k": "Holnapi meccsek \u00e9s predikci\u00f3k",
    "Holnapi meccsek ?cs predikci?k": "Holnapi meccsek \u00e9s predikci\u00f3k",
    "Adatok bet?\u0014lt?cse...": "Adatok bet\u00f6lt\u00e9se...",
    "Fogad?iroda:": "Fogad\u00f3iroda:",
    "Keres?cs:": "Keres\u00e9s:",
    "Friss??t?cs": "Friss\u00edt\u00e9s",
    "D?tum": "D\u00e1tum",
    "K?csz.": "K\u00e9sz.",
    "Hiba t?\u0014rt?cnt.": "Hiba t\u00f6rt\u00e9nt.",
}

# Global process-level throttles/state
_RESULT_CHECKED_AT: Dict[int, float] = {}
_LAST_RETRY_PRED_AT: float | None = None


def _fix_mojibake(s: str) -> str:
    try:
        out = s
        for bad, good in _MOJIBAKE_MAP.items():
            if bad in out:
                out = out.replace(bad, good)
        # Heuristic re-decode for common mojibake (cp1252/latin-1 seen as UTF-8)
        accented = "\u00e1\u00e9\u00ed\u00f3\u00f6\u0151\u00fa\u00fc\u0171\u00c1\u00c9\u00cd\u00d3\u00d6\u0150\u00da\u00dc\u0170"
        if "�" in out or any(ch in out for ch in ("Ã", "Â", "��")):
            for enc in ("cp1252", "latin-1"):
                try:
                    cand = out.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
                    # Prefer candidate if it removes replacement chars or adds accented letters
                    if ("�" in out and "�" not in cand) or any(ch in cand for ch in accented):
                        out = cand
                        break
                except Exception:
                    continue
        return out
    except Exception:
        return s


def _normalize_widget_texts(container: tk.Misc) -> None:
    try:
        for w in container.winfo_children():
            try:
                txt = cast(Any, w).cget("text")
            except Exception:
                txt = None
            if isinstance(txt, str):
                new_txt = _fix_mojibake(txt)
                if new_txt != txt:
                    try:
                        cast(Any, w).configure(text=new_txt)
                    except Exception:
                        pass
            # Recurse into frames and other containers
            _normalize_widget_texts(w)
    except Exception:
        pass


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    need_migrate = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    if need_migrate:
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
    return conn


def _open_ro() -> sqlite3.Connection:
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


def _tomorrow_budapest_date_str() -> str:
    try:
        from zoneinfo import ZoneInfo

        now_bud = datetime.now(ZoneInfo("Europe/Budapest"))
        return (now_bud.date() + timedelta(days=1)).isoformat()
    except Exception:
        return (datetime.now().date() + timedelta(days=1)).isoformat()


def _select_fixtures_for_tomorrow() -> List[Mapping[str, Any]]:
    leagues = LeaguesService().get_current_leagues()
    fx_svc = FixturesService()
    odds_svc = OddsService()
    day = _tomorrow_budapest_date_str()

    fixtures: List[Mapping[str, Any]] = []
    for ls in leagues:
        league_id = int(ls["league_id"])
        season = int(ls["season_year"])
        rows = fx_svc.get_daily_fixtures(day, league_id=league_id, season=season)
        for r in rows:
            fx_id = r.get("fixture_id")
            if not isinstance(fx_id, int):
                continue
            try:
                if odds_svc.get_fixture_odds(fx_id):
                    fixtures.append(r)
            except Exception:
                continue
    return fixtures


def _ensure_league(conn: sqlite3.Connection, league_id: int) -> None:
    lrepo = LeaguesRepoSqlite(conn)
    if lrepo.get_by_id(league_id) is None:
        try:
            lrepo.insert(League(id=league_id, name=f"League {league_id}", country=None))
        except Exception:
            pass


def _predict_then_persist_if_complete(conn: sqlite3.Connection, fixture: Mapping[str, Any]) -> bool:
    """Compute predictions for a fixture and persist only if all models predicted.

    - Builds domain match and model context from the fixture metadata
    - Runs all default models
    - If any model returns no probabilities (SKIPPED), nothing is saved
    - If all models return probabilities, saves: league (if needed), match, odds, predictions
    Returns True if persisted, False otherwise.
    """
    try:
        fx_id_val = fixture.get("fixture_id")
        league_id_val = fixture.get("league_id")
        if not isinstance(fx_id_val, int) or not isinstance(league_id_val, int):
            return False
        fx_id = int(fx_id_val)
        league_id = int(league_id_val)
        season = int(fixture.get("season") or datetime.now().year)
        home_id = int(fixture.get("home_id") or 0)
        away_id = int(fixture.get("away_id") or 0)
        when = fixture.get("date_utc")
        if not isinstance(when, datetime):
            when = datetime.now(timezone.utc)

        # Check if all predictions already exist in DB for this match
        try:
            models = default_models()
            expected_models: set[str] = set()
            for m in models:
                try:
                    expected_models.add(str(m.name.value))
                except Exception:
                    expected_models.add(str(m.name))
            cur = conn.execute(
                "SELECT DISTINCT model_name FROM predictions WHERE match_id = ?",
                (fx_id,),
            )
            present = {str(r[0]) for r in cur.fetchall()}
            if expected_models and expected_models.issubset(present):
                return True
        except Exception:
            pass

        # Domain match + modeling context
        status_raw = fixture.get("status")
        try:
            status = (
                MatchStatus(status_raw) if isinstance(status_raw, str) else MatchStatus.SCHEDULED
            )
        except Exception:
            status = MatchStatus.SCHEDULED

        match = DomainMatch(
            fixture_id=FixtureId(fx_id),
            league_id=LeagueId(league_id),
            season=season,
            kickoff_utc=when,
            home_name=str(fixture.get("home_name") or fixture.get("home_id") or ""),
            away_name=str(fixture.get("away_name") or fixture.get("away_id") or ""),
            status=status,
        )

        history_obj: Any
        try:
            history_obj = HistoryService()
        except Exception:
            # Lightweight fallback that satisfies the ContextBuilder history protocol
            class _HistoryFallback:
                def get_recent_team_scores(
                    self,
                    team_id: int,
                    league_id: int,
                    season: int,
                    last: int,
                    *,
                    only_finished: bool = True,
                ) -> list[dict[str, Any]]:
                    return []

                def league_goal_means(self, league_id: int, season: int) -> tuple[float, float]:
                    return (1.35, 1.15)

                def get_team_averages(self, team_id: int, league_id: int, season: int) -> Any:
                    class _TA:
                        goals_for_home_avg = 1.3
                        goals_for_away_avg = 1.1
                        goals_against_home_avg = 1.2
                        goals_against_away_avg = 1.2

                    return _TA()

                def simple_poisson_means(self, home: Any, away: Any) -> tuple[float, float]:
                    mu_h = (
                        float(getattr(home, "goals_for_home_avg", 1.3))
                        + float(getattr(away, "goals_against_away_avg", 1.2))
                    ) / 2.0
                    mu_a = (
                        float(getattr(away, "goals_for_away_avg", 1.1))
                        + float(getattr(home, "goals_against_home_avg", 1.2))
                    ) / 2.0
                    return (mu_h, mu_a)

            history_obj = _HistoryFallback()
        # Compute features only if a model that needs them is present (e.g., logistic_regression)
        need_feats = False
        try:
            for m in models:
                name_str = str(getattr(m.name, "value", m.name))
                if name_str == "logistic_regression":
                    need_feats = True
                    break
        except Exception:
            need_feats = True
        try:
            ctx_builder = ContextBuilder(history=history_obj, compute_features=bool(need_feats))
        except TypeError:
            # Backward/monkeypatch compatibility: some tests replace ContextBuilder
            # with a simplified fake that doesn't accept compute_features.
            ctx_builder = ContextBuilder(history=history_obj)
        ctx = ctx_builder.build_from_meta(
            fixture_id=fx_id,
            league_id=league_id,
            season=season,
            home_team_id=home_id,
            away_team_id=away_id,
        )

        agg = PredictionAggregatorImpl()
        preds = agg.run_all(models, match, ctx)

        # All models must produce probabilities (no SKIPPED)
        if any(p.probs is None for p in preds):
            return False

        # Ensure league exists only now (since we will persist)
        _ensure_league(conn, league_id)

        # Insert match if not exists
        mrepo = MatchesRepoSqlite(conn)
        if mrepo.get_by_id(fx_id) is None:
            db_match = DbMatch(
                id=fx_id,
                league_id=league_id,
                season=season,
                date=when,
                home_team=str(fixture.get("home_name") or fixture.get("home_id") or ""),
                away_team=str(fixture.get("away_name") or fixture.get("away_id") or ""),
                real_result=None,
            )
            mrepo.insert(db_match)

        # Persist odds for this fixture (and bookmaker names)
        try:
            svc = OddsService()
            orepo = OddsRepoSqlite(conn)
            brepo = BookmakersRepoSqlite(conn)
            odds_list = svc.get_fixture_odds(int(fx_id))
            bm_names = svc.get_fixture_bookmakers(int(fx_id))
            for bid, name in bm_names.items():
                try:
                    if brepo.get_by_id(int(bid)) is None:
                        brepo.insert(Bookmaker(id=int(bid), name=str(name)))
                except Exception:
                    pass
            for o in odds_list:
                try:
                    orepo.insert(
                        RepoOdds(
                            id=None,
                            match_id=int(getattr(o, "fixture_id", fx_id)),
                            bookmaker_id=int(getattr(o, "bookmaker_id")),
                            home=float(getattr(o, "home")),
                            draw=float(getattr(o, "draw")),
                            away=float(getattr(o, "away")),
                        )
                    )
                except Exception:
                    continue
        except Exception:
            # Odds persistence issues are tolerated; match+preds already saved
            pass

        # Persist predictions
        prepo = PredictionsRepoSqlite(conn)
        for p in preds:
            pr = p.probs
            if pr is None:
                continue
            triple = [float(pr.home), float(pr.draw), float(pr.away)]
            idx = int(max(range(3), key=lambda i: triple[i]))
            pred_res = "1" if idx == 0 else ("X" if idx == 1 else "2")

            dp = DbPrediction(
                id=None,
                match_id=fx_id,
                model_name=str(p.model.value if hasattr(p.model, "value") else p.model),
                prob_home=float(pr.home),
                prob_draw=float(pr.draw),
                prob_away=float(pr.away),
                predicted_result=pred_res,
                is_correct=None,
                result_status="PENDING",
            )
            prepo.insert(dp)

        return True
    except Exception:
        return False
    """Compute predictions for a fixture and persist only if all models predicted.

    - Builds domain match and model context from the fixture metadata
    - Runs all default models
    - If any model returns no probabilities (SKIPPED), nothing is saved
    - If all models return probabilities, saves: league (if needed), match, odds, predictions
    Returns True if persisted, False otherwise.
    """
    try:
        fx_id_val = fixture.get("fixture_id")
        league_id_val = fixture.get("league_id")
        if not isinstance(fx_id_val, int) or not isinstance(league_id_val, int):
            return False
        fx_id = int(fx_id_val)
        league_id = int(league_id_val)
        season = int(fixture.get("season") or datetime.now().year)
        home_id = int(fixture.get("home_id") or 0)
        away_id = int(fixture.get("away_id") or 0)
        when = fixture.get("date_utc")
        if not isinstance(when, datetime):
            when = datetime.now(timezone.utc)

        # Check if all predictions already exist in DB for this match
        try:
            models = default_models()
            expected_models_dup: set[str] = set()
            for m in models:
                try:
                    expected_models_dup.add(str(m.name.value))
                except Exception:
                    expected_models_dup.add(str(m.name))
            cur = conn.execute(
                "SELECT DISTINCT model_name FROM predictions WHERE match_id = ?",
                (fx_id,),
            )
            present = {str(r[0]) for r in cur.fetchall()}
            if expected_models_dup and expected_models_dup.issubset(present):
                return True
        except Exception:
            pass

        # Domain match + modeling context
        status_raw = fixture.get("status")
        try:
            status = (
                MatchStatus(status_raw) if isinstance(status_raw, str) else MatchStatus.SCHEDULED
            )
        except Exception:
            status = MatchStatus.SCHEDULED

        match = DomainMatch(
            fixture_id=FixtureId(fx_id),
            league_id=LeagueId(league_id),
            season=season,
            kickoff_utc=when,
            home_name=str(fixture.get("home_name") or fixture.get("home_id") or ""),
            away_name=str(fixture.get("away_name") or fixture.get("away_id") or ""),
            status=status,
        )

        history = HistoryService()
        # Compute features only if a model that needs them is present (e.g., logistic_regression)
        need_feats = False
        try:
            for m in models:
                name_str = str(getattr(m.name, "value", m.name))
                if name_str == "logistic_regression":
                    need_feats = True
                    break
        except Exception:
            need_feats = True
        ctx_builder = ContextBuilder(history=history, compute_features=bool(need_feats))
        ctx = ctx_builder.build_from_meta(
            fixture_id=fx_id,
            league_id=league_id,
            season=season,
            home_team_id=home_id,
            away_team_id=away_id,
        )

        agg = PredictionAggregatorImpl()
        preds = agg.run_all(models, match, ctx)

        # All models must produce probabilities (no SKIPPED)
        if any(p.probs is None for p in preds):
            return False

        # Ensure league exists only now (since we will persist)
        _ensure_league(conn, league_id)

        # Insert match if not exists
        mrepo = MatchesRepoSqlite(conn)
        if mrepo.get_by_id(fx_id) is None:
            m = DbMatch(
                id=fx_id,
                league_id=league_id,
                season=season,
                date=when,
                home_team=str(fixture.get("home_name") or fixture.get("home_id") or ""),
                away_team=str(fixture.get("away_name") or fixture.get("away_id") or ""),
                real_result=None,
            )
            mrepo.insert(m)

        # Persist odds for this fixture (and bookmaker names)
        try:
            svc = OddsService()
            orepo = OddsRepoSqlite(conn)
            brepo = BookmakersRepoSqlite(conn)
            odds_list = svc.get_fixture_odds(int(fx_id))
            bm_names = svc.get_fixture_bookmakers(int(fx_id))
            for bid, name in bm_names.items():
                try:
                    if brepo.get_by_id(int(bid)) is None:
                        brepo.insert(Bookmaker(id=int(bid), name=str(name)))
                except Exception:
                    pass
            for o in odds_list:
                try:
                    orepo.insert(
                        RepoOdds(
                            id=None,
                            match_id=int(getattr(o, "fixture_id", fx_id)),
                            bookmaker_id=int(getattr(o, "bookmaker_id")),
                            home=float(getattr(o, "home")),
                            draw=float(getattr(o, "draw")),
                            away=float(getattr(o, "away")),
                        )
                    )
                except Exception:
                    continue
        except Exception:
            # Odds persistence issues are tolerated; match+preds already saved
            pass

        # Persist predictions
        prepo = PredictionsRepoSqlite(conn)
        for p in preds:
            pr = p.probs
            if pr is None:
                continue
            triple = [float(pr.home), float(pr.draw), float(pr.away)]
            idx = int(max(range(3), key=lambda i: triple[i]))
            pred_res = "1" if idx == 0 else ("X" if idx == 1 else "2")

            dp = DbPrediction(
                id=None,
                match_id=fx_id,
                model_name=str(p.model.value if hasattr(p.model, "value") else p.model),
                prob_home=float(pr.home),
                prob_draw=float(pr.draw),
                prob_away=float(pr.away),
                predicted_result=pred_res,
                is_correct=None,
                result_status="PENDING",
            )
            prepo.insert(dp)

        return True
    except Exception:
        return False

    # Legacy helper _predict_and_store removed; replaced by per-fixture
    # _predict_then_persist_if_complete.


def _read_tomorrow_from_db(
    conn: sqlite3.Connection,
) -> Tuple[List[Tuple], Dict[int, Dict[str, Tuple[float, float, float, str | None]]]]:
    cur = conn.cursor()
    today_utc = datetime.now(timezone.utc).date()
    start_utc = datetime.combine(
        today_utc + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    end_utc = start_utc + timedelta(days=1)

    # Compare ISO-8601 strings directly; stored values are ISO with timezone (+00:00),
    # and bounds are generated in the same format. Avoid SQLite datetime() which
    # does not parse "+00:00" offsets reliably.
    cur.execute(
        """
        SELECT match_id, date, home_team, away_team
        FROM matches
        WHERE date >= ? AND date < ?
        ORDER BY date ASC
        """,
        (start_utc.isoformat(), end_utc.isoformat()),
    )
    matches = cur.fetchall()

    preds_by_match: Dict[int, Dict[str, Tuple[float, float, float, str | None]]] = {}
    for mid, _dt, _h, _a in matches:
        cur.execute(
            """
            SELECT model_name, prob_home, prob_draw, prob_away, predicted_result
            FROM predictions
            WHERE match_id = ?
            """,
            (mid,),
        )
        rows = cur.fetchall()
        preds_by_match[int(mid)] = {
            str(model): (float(ph), float(pd), float(pa), (pred if isinstance(pred, str) else None))
            for (model, ph, pd, pa, pred) in rows
        }

    return matches, preds_by_match


def _read_played_from_db(
    conn: sqlite3.Connection, *, days: int = 7
) -> Tuple[List[Tuple], Dict[int, Dict[str, Tuple[float, float, float, str | None]]]]:
    """Return finished matches within the last `days` days with predictions.

    Uses status via persisted `real_result` instead of only date.
    """
    cur = conn.cursor()
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=int(days))
    cur.execute(
        """
        SELECT match_id, date, home_team, away_team
        FROM matches
        WHERE real_result IN ('1','X','2') AND date < ? AND date >= ?
        ORDER BY date DESC
        """,
        (now_utc.isoformat(), start_utc.isoformat()),
    )
    matches = cur.fetchall()

    preds_by_match: Dict[int, Dict[str, Tuple[float, float, float, str | None]]] = {}
    for mid, _dt, _h, _a in matches:
        cur.execute(
            """
            SELECT model_name, prob_home, prob_draw, prob_away, predicted_result
            FROM predictions
            WHERE match_id = ?
            """,
            (mid,),
        )
        rows = cur.fetchall()
        preds_by_match[int(mid)] = {
            str(model): (float(ph), float(pd), float(pa), (pred if isinstance(pred, str) else None))
            for (model, ph, pd, pa, pred) in rows
        }

    return matches, preds_by_match


def _update_missing_results(conn: sqlite3.Connection) -> None:
    """Fetch and persist real results for past matches missing `real_result`.

    For each match with date < now and NULL real_result, query API for result
    and persist '1'/'X'/'2' when available.
    """
    try:
        # Use a separate write connection to avoid read-only GUI connection
        try:
            wconn = _ensure_db()
        except Exception:
            wconn = conn
        # Throttle per-fixture result checks within this process
        global _RESULT_CHECKED_AT
        try:
            _RESULT_CHECKED_AT
        except NameError:
            _RESULT_CHECKED_AT = {}
        cur = wconn.execute(
            "SELECT match_id FROM matches WHERE real_result IS NULL AND date < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        rows = cur.fetchall()
        if not rows:
            return
        history = HistoryService()
        mrepo = MatchesRepoSqlite(wconn)
        for (mid,) in rows:
            try:
                last_ts = (
                    _RESULT_CHECKED_AT.get(int(mid))
                    if isinstance(_RESULT_CHECKED_AT, dict)
                    else None
                )
                if last_ts is not None and (time.time() - float(last_ts)) < 3600.0:
                    continue
                label = history.get_fixture_result_label(int(mid))
                if label in {"1", "X", "2"}:
                    mrepo.update_result(int(mid), str(label))
                _RESULT_CHECKED_AT[int(mid)] = time.time()
            except Exception:
                continue
    except Exception:
        pass


def _reconcile_prediction_statuses(conn: sqlite3.Connection) -> None:
    """Mark predictions WIN/LOSE where `matches.real_result` is known.

    This is a pure DB-side reconciliation: no network calls.
    """
    try:
        conn.execute(
            """
            UPDATE predictions
            SET
                is_correct = CASE
                    WHEN predicted_result = (
                        SELECT real_result FROM matches m WHERE m.match_id = predictions.match_id
                    ) THEN 1 ELSE 0 END,
                result_status = CASE
                    WHEN predicted_result = (
                        SELECT real_result FROM matches m WHERE m.match_id = predictions.match_id
                    ) THEN 'WIN' ELSE 'LOSE' END
            WHERE match_id IN (
                SELECT match_id FROM matches WHERE real_result IN ('1','X','2')
            )
            """
        )
        conn.commit()
    except Exception:
        pass


# ---------------------------- GUI ----------------------------

MODEL_HEADERS = {
    "poisson": "Poisson",
    "monte_carlo": "MonteCarlo",
    "elo": "Elo",
    "logistic_regression": "LogReg",
    "balance": "Balance",
    "veto": "Veto",
}


@dataclass
class AppState:
    conn: sqlite3.Connection
    tree: ttk.Treeview
    selected_bookmaker_id: int | None = None
    selected_model_key: str | None = None
    bm_names_by_id: Dict[int, str] | None = None
    bm_combo_var: tk.StringVar | None = None
    bm_combo: ttk.Combobox | None = None
    model_combo_var: tk.StringVar | None = None
    model_combo: ttk.Combobox | None = None
    search_var: tk.StringVar | None = None
    sort_by: int | None = None
    sort_reverse: bool = False
    # View state
    show_played: bool = False
    winners_only: bool = False
    winners_only_var: tk.BooleanVar | None = None
    winners_only_chk: ttk.Checkbutton | None = None
    # Stats side panel
    stats_frame: Any | None = None
    stats_vars: Dict[str, tk.StringVar] | None = None


def _update_model_stats_panel(
    state: AppState,
    filtered_matches: list[Tuple[Any, ...]],
    preds_by_match: Dict[int, Dict[str, Tuple[float, float, float, str | None]]],
) -> None:
    try:
        # Only show in played view with a single model selected
        if not getattr(state, "show_played", False) or state.selected_model_key is None:
            if state.stats_frame is not None:
                try:
                    state.stats_frame.grid_remove()
                except Exception:
                    pass
            return

        # Ensure side panel exists
        if state.stats_frame is None or state.stats_vars is None:
            return

        model_key = state.selected_model_key
        use_bm = state.selected_bookmaker_id is not None

        total = 0
        wins = 0
        used = 0
        sum_odds = 0.0
        sum_ev = 0.0
        pos_ev_win = 0

        for mid, _dt_s, _home, _away in filtered_matches:
            try:
                try:
                    mid_i = int(mid)
                except Exception:
                    continue
                model_map = preds_by_match.get(mid_i, {})
                trip = model_map.get(model_key) if isinstance(model_map, dict) else None
                if not trip:
                    continue
                ph, pd, pa, pref = trip
                if isinstance(pref, str) and pref in {"1", "X", "2"}:
                    sel = pref
                else:
                    vals = [("1", float(ph)), ("X", float(pd)), ("2", float(pa))]
                    sel, _ = max(vals, key=lambda x: x[1])

                # Real result
                rr = None
                try:
                    rrow = state.conn.execute(
                        "SELECT real_result FROM matches WHERE match_id = ?",
                        (mid_i,),
                    ).fetchone()
                    if rrow and isinstance(rrow[0], str):
                        rr = rrow[0]
                except Exception:
                    rr = None
                if rr not in {"1", "X", "2"}:
                    continue

                # Determine odds for the selected outcome
                odd_val = None
                if use_bm:
                    try:
                        row = state.conn.execute(
                            "SELECT odds_home, odds_draw, odds_away FROM odds WHERE match_id = ? AND bookmaker_id = ? LIMIT 1",
                            (mid_i, cast(int, state.selected_bookmaker_id)),
                        ).fetchone()
                        if row is not None:
                            oh, od, oa = float(row[0]), float(row[1]), float(row[2])
                            odd_val = oh if sel == "1" else (od if sel == "X" else oa)
                    except Exception:
                        pass
                else:
                    try:
                        best: Dict[str, float] = {}
                        for bid_raw, oh, od, oa in state.conn.execute(
                            "SELECT bookmaker_id, odds_home, odds_draw, odds_away FROM odds WHERE match_id = ?",
                            (mid_i,),
                        ).fetchall():
                            ohf, odf, oaf = float(oh), float(od), float(oa)
                            if ("1" not in best) or (ohf > best["1"]):
                                best["1"] = ohf
                            if ("X" not in best) or (odf > best["X"]):
                                best["X"] = odf
                            if ("2" not in best) or (oaf > best["2"]):
                                best["2"] = oaf
                        if sel in best:
                            odd_val = best[sel]
                    except Exception:
                        pass

                total += 1
                if rr == sel:
                    wins += 1

                # If we have odds and probabilities, compute EV
                p = float(ph) if sel == "1" else (float(pd) if sel == "X" else float(pa))
                if odd_val is not None:
                    used += 1
                    sum_odds += float(odd_val)
                    ev = p * float(odd_val) - 1.0
                    sum_ev += ev
                    if ev > 0 and rr == sel:
                        pos_ev_win += 1
            except Exception:
                continue

        # Compute metrics
        hit_rate = (wins / total * 100.0) if total else 0.0
        avg_odds = (sum_odds / used) if used else 0.0
        avg_ev = (sum_ev / used * 100.0) if used else 0.0

        if state.stats_vars is not None:
            v = state.stats_vars.get("count")
            if v is not None:
                v.set(str(total))
            v = state.stats_vars.get("hit")
            if v is not None:
                v.set(f"{hit_rate:.0f}% ({wins}/{total})")
            v = state.stats_vars.get("avg_odds")
            if v is not None:
                v.set(f"{avg_odds:.2f}" if used else "-")
            v = state.stats_vars.get("avg_ev")
            if v is not None:
                v.set(f"{avg_ev:+.0f}%" if used else "-")
            v = state.stats_vars.get("pos_ev_win")
            if v is not None:
                v.set(str(pos_ev_win))

        # Show the panel
        try:
            state.stats_frame.grid()
        except Exception:
            pass
    except Exception:
        # On any error, hide panel
        try:
            if state.stats_frame is not None:
                state.stats_frame.grid_remove()
        except Exception:
            pass


def _best_label_and_pct(
    ph: float, pd: float, pa: float, preferred: str | None = None
) -> Tuple[str, float]:
    if preferred in {"1", "X", "2"}:
        lbl = preferred
        pct = {"1": ph, "X": pd, "2": pa}[lbl] * 100.0
        return lbl, pct
    vals = [("1", ph), ("X", pd), ("2", pa)]
    lbl, v = max(vals, key=lambda x: x[1])
    return lbl, v * 100.0


def refresh_table(state: AppState) -> None:
    for i in state.tree.get_children():
        state.tree.delete(i)

    # Update missing results and reconcile statuses before loading rows
    try:
        _update_missing_results(state.conn)
        _reconcile_prediction_statuses(state.conn)
    except Exception:
        pass
    # Opportunistic retry: for upcoming view, attempt to process fixtures not yet persisted
    try:
        if not getattr(state, "show_played", False):
            global _LAST_RETRY_PRED_AT
            import time as _t

            last = _LAST_RETRY_PRED_AT
            # Retry at most every 300s
            if last is None or (_t.time() - float(last)) > 300.0:
                try:
                    write_conn = _ensure_db()
                except Exception:
                    write_conn = state.conn
                fixtures = _select_fixtures_for_tomorrow()
                for r in fixtures:
                    try:
                        _predict_then_persist_if_complete(write_conn, r)
                    except Exception:
                        pass
                _LAST_RETRY_PRED_AT = _t.time()
    except Exception:
        pass
    if getattr(state, "show_played", False):
        matches, preds_by_match = _read_played_from_db(state.conn, days=7)
    else:
        matches, preds_by_match = _read_tomorrow_from_db(state.conn)

    # Apply bookmaker filter: None -> only matches with any odds; otherwise only those with a row for selected bookmaker
    def _has_any_odds(mid: int) -> bool:
        cur = state.conn.execute("SELECT 1 FROM odds WHERE match_id = ? LIMIT 1", (mid,))
        return cur.fetchone() is not None

    def _has_bookmaker(mid: int, bid: int) -> bool:
        cur = state.conn.execute(
            "SELECT 1 FROM odds WHERE match_id = ? AND bookmaker_id = ? LIMIT 1", (mid, bid)
        )
        return cur.fetchone() is not None

    # Apply free-text team filter if provided
    q = (state.search_var.get() if state.search_var is not None else "").strip().lower()

    filtered: list[Tuple[Any, ...]] = []
    for mid, dt_s, home, away in matches:
        # Text filter on "home - away"
        if q:
            label = f"{home} - {away}".lower()
            if q not in label:
                continue
        if state.selected_bookmaker_id is None:
            if not _has_any_odds(int(mid)):
                continue
        else:
            if not _has_bookmaker(int(mid), int(state.selected_bookmaker_id)):
                continue
        # If a specific model is selected, keep only matches with that prediction present
        if state.selected_model_key is not None:
            model_map_check = preds_by_match.get(int(mid), {})
            if state.selected_model_key not in model_map_check:
                continue
        filtered.append((mid, dt_s, home, away))

    for mid, dt_s, home, away in filtered:
        try:
            dt = datetime.fromisoformat(dt_s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            try:
                from zoneinfo import ZoneInfo

                disp_dt = dt.astimezone(ZoneInfo("Europe/Budapest")).strftime("%Y-%m-%d %H:%M")
            except Exception:
                disp_dt = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            disp_dt = str(dt_s)

        model_map = preds_by_match.get(int(mid), {})
        # Fetch real result for display (1/X/2) if available
        result_str = ""
        try:
            cur_res = state.conn.execute(
                "SELECT real_result FROM matches WHERE match_id = ?",
                (int(mid),),
            )
            rres = cur_res.fetchone()
            if rres is not None and isinstance(rres[0], str):
                result_str = str(rres[0])
        except Exception:
            result_str = ""
        # Winners-only filter: only in played view with a selected model
        try:
            if (
                getattr(state, "show_played", False)
                and getattr(state, "winners_only", False)
                and state.selected_model_key is not None
            ):
                if result_str not in {"1", "X", "2"}:
                    continue
                trip_sel_f = model_map.get(state.selected_model_key)
                if not trip_sel_f:
                    continue
                phf, pdf, paf, preff = trip_sel_f
                if isinstance(preff, str) and preff in {"1", "X", "2"}:
                    sel_lbl_f = preff
                else:
                    vals_f = [("1", float(phf)), ("X", float(pdf)), ("2", float(paf))]
                    sel_lbl_f, _ = max(vals_f, key=lambda x: x[1])
                if sel_lbl_f != result_str:
                    continue
        except Exception:
            pass
        values = [disp_dt, f"{home} - {away}", result_str]
        # If a bookmaker is selected, pull odds once per match to render beneath the probabilities
        match_odds: Tuple[float, float, float] | None = None
        if state.selected_bookmaker_id is not None:
            try:
                cur = state.conn.execute(
                    "SELECT odds_home, odds_draw, odds_away FROM odds WHERE match_id = ? AND bookmaker_id = ? LIMIT 1",
                    (int(mid), int(state.selected_bookmaker_id)),
                )
                r_any = cur.fetchone()
                if r_any is not None:
                    r_t = cast(Tuple[float, float, float], r_any)
                    match_odds = (float(r_t[0]), float(r_t[1]), float(r_t[2]))
            except Exception:
                match_odds = None

        for key in MODEL_HEADERS.keys():
            trip = model_map.get(key)
            if trip:
                ph, pd, pa, pref = trip
                lbl, pct = _best_label_and_pct(ph, pd, pa, pref)
                values.append(f"{lbl} ({pct:.0f}%)")
            else:
                values.append("-")
        # Add odds shopping column when a single model is selected
        ev_cell = ""
        if state.selected_model_key is not None:
            best_text = ""
            try:
                trip_sel = model_map.get(state.selected_model_key)
                if trip_sel:
                    ph, pd, pa, pref = trip_sel
                    if isinstance(pref, str) and pref in {"1", "X", "2"}:
                        sel_lbl = pref
                    else:
                        vals = [("1", ph), ("X", pd), ("2", pa)]
                        sel_lbl, _ = max(vals, key=lambda x: x[1])
                    # If both model and bookmaker are selected, show that bookmaker's odd for the tip
                    if state.selected_bookmaker_id is not None and match_odds is not None:
                        oh, od, oa = match_odds
                        odd_val = oh if sel_lbl == "1" else (od if sel_lbl == "X" else oa)
                        best_text = f"{odd_val:.2f}"
                        # Compute expected value for selected bookmaker and model tip
                        p = ph if sel_lbl == "1" else (pd if sel_lbl == "X" else pa)
                        ev = p * (odd_val) - 1.0
                        ev_cell = f"{ev*100:+.0f}%"
                    else:
                        # Otherwise compute and show best across bookmakers for the tip
                        best: Dict[str, Tuple[int, float]] = {}
                        cur2 = state.conn.execute(
                            "SELECT bookmaker_id, odds_home, odds_draw, odds_away FROM odds WHERE match_id = ?",
                            (int(mid),),
                        )
                        for bid_raw, oh, od, oa in cur2.fetchall():
                            bid = int(bid_raw)
                            ohf, odf, oaf = float(oh), float(od), float(oa)
                            if ("1" not in best) or (ohf > best["1"][1]):
                                best["1"] = (bid, ohf)
                            if ("X" not in best) or (odf > best["X"][1]):
                                best["X"] = (bid, odf)
                            if ("2" not in best) or (oaf > best["2"][1]):
                                best["2"] = (bid, oaf)
                        if sel_lbl in best:
                            bid, odd = best[sel_lbl]
                            if state.bm_names_by_id:
                                bm_name = state.bm_names_by_id.get(int(bid), str(bid))
                            else:
                                bm_name = str(bid)
                            best_text = f"{bm_name} {odd:.2f}"
                            # Compute BEST EV when no bookmaker selected
                            p = ph if sel_lbl == "1" else (pd if sel_lbl == "X" else pa)
                            ev = p * odd - 1.0
                            ev_cell = f"{ev*100:+.0f}%"
            except Exception:
                best_text = ""
            values.append(best_text)
            # Append EV cell right after the tip-odds column
            values.append(ev_cell)
        # If BEST_ODDS_COL exists but no single model selected, pad empty cells to keep columns aligned
        try:
            if state.selected_model_key is None:
                cols_tuple = cast(Tuple[str, ...], state.tree["columns"])
                if any(c == "Best odds" for c in cols_tuple):
                    values.append("")
                # EV column placeholder
                if any(c == "EV" for c in cols_tuple):
                    values.append("")
        except Exception:
            pass
        # Append detailed odds columns only when bookmaker is selected AND no specific model is selected
        if state.selected_bookmaker_id is not None and state.selected_model_key is None:
            if match_odds is not None:
                oh, od, oa = match_odds
                values.extend([f"{oh:.2f}", f"{od:.2f}", f"{oa:.2f}"])
            else:
                values.extend(["", "", ""])
        # use match_id as iid for easy retrieval on double-click
        # Zebra striping for readability
        tag = "odd" if (len(state.tree.get_children()) % 2 == 1) else "even"
        # Row coloring for played matches when a single model is selected
        row_tags = [tag]
        try:
            if getattr(state, "show_played", False) and state.selected_model_key is not None:
                if result_str in {"1", "X", "2"}:
                    trip_sel2 = model_map.get(state.selected_model_key)
                    if trip_sel2:
                        ph2, pd2, pa2, pref2 = trip_sel2
                        if isinstance(pref2, str) and pref2 in {"1", "X", "2"}:
                            sel_lbl2 = pref2
                        else:
                            vals2 = [("1", float(ph2)), ("X", float(pd2)), ("2", float(pa2))]
                            sel_lbl2, _ = max(vals2, key=lambda x: x[1])
                        if sel_lbl2 == result_str:
                            row_tags.append("row_win")
                        else:
                            row_tags.append("row_lose")
        except Exception:
            pass
        state.tree.insert("", "end", iid=str(int(mid)), values=values, tags=tuple(row_tags))

    # Update stats side panel if applicable
    try:
        _update_model_stats_panel(state, filtered, preds_by_match)
    except Exception:
        pass


def _show_odds_window(tree: ttk.Treeview, fixture_id: int) -> None:
    try:
        svc = OddsService()
        odds_list = svc.get_fixture_odds(fixture_id)
        names = svc.get_fixture_bookmakers(fixture_id)
    except Exception as exc:
        messagebox.showerror("Hiba", f"Odds lek\u00e9r\u00e9se sikertelen: {exc}")
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds - {fixture_id}")
    top.geometry("600x400")
    try:
        top.title(f"Odds - {fixture_id}")
    except Exception:
        pass
    cols = ["Bookmaker", "Home", "Draw", "Away"]
    tv = ttk.Treeview(top, columns=cols, show="headings")
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=120 if c != "Bookmaker" else 200, anchor=tk.W)
    vsb = ttk.Scrollbar(top, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    tv.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    top.grid_rowconfigure(0, weight=1)
    top.grid_columnconfigure(0, weight=1)
    try:
        _normalize_widget_texts(top)
    except Exception:
        pass

    # Populate
    for o in odds_list:
        bm_name = (
            names.get(int(o.bookmaker_id), str(o.bookmaker_id))
            if hasattr(o, "bookmaker_id")
            else "?"
        )
        try:
            home = f"{o.home:.2f}"
            draw = f"{o.draw:.2f}"
            away = f"{o.away:.2f}"
        except Exception:
            home = draw = away = "?"
    tv.insert("", "end", values=[bm_name, home, draw, away])


def _show_odds_window_db(conn: sqlite3.Connection, tree: ttk.Treeview, fixture_id: int) -> None:
    try:
        from src.repositories.sqlite.bookmakers_sqlite import BookmakersRepoSqlite
        from src.repositories.sqlite.odds_sqlite import OddsRepoSqlite

        orepo = OddsRepoSqlite(conn)
        brepo = BookmakersRepoSqlite(conn)
        odds_list = orepo.list_by_match(int(fixture_id))
        names: Dict[int, str] = {}
        for o in odds_list:
            bid = int(o.bookmaker_id)
            if bid not in names:
                b = brepo.get_by_id(bid)
                names[bid] = b.name if b else str(bid)
    except Exception as exc:
        messagebox.showerror(
            "Hiba", f"Odds bet\u00f6lt\u00e9se az adatb\u00e1zisb\u00f3l sikertelen: {exc}"
        )
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds - {fixture_id}")
    top.geometry("600x400")
    try:
        top.title(f"Odds - {fixture_id}")
    except Exception:
        pass
    cols = ["Bookmaker", "Home", "Draw", "Away"]
    tv = ttk.Treeview(top, columns=cols, show="headings")
    for c in cols:
        tv.heading(c, text=c)
        tv.column(c, width=120 if c != "Bookmaker" else 200, anchor=tk.W)
    vsb = ttk.Scrollbar(top, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=vsb.set)
    tv.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    top.grid_rowconfigure(0, weight=1)
    top.grid_columnconfigure(0, weight=1)
    try:
        _normalize_widget_texts(top)
    except Exception:
        pass

    for o in odds_list:
        bid = int(o.bookmaker_id)
        bm_name = names.get(bid, str(bid))
        home = f"{float(o.home):.2f}"
        draw = f"{float(o.draw):.2f}"
        away = f"{float(o.away):.2f}"
        tv.insert("", "end", values=[bm_name, home, draw, away])

    # Removed old full-coverage helpers: _tomorrow_bounds_iso, _get_tomorrow_match_ids,
    # _models_for_match, _default_model_names


def run_app() -> None:
    # Build GUI first so errors can be shown reliably
    root = tk.Tk()
    root.title("Holnapi meccsek \u00e9s predikci\u00f3k")
    root.geometry("1200x600")
    # Ensure proper accented title regardless of source encoding
    try:
        root.title("Holnapi meccsek \u00e9s predikci\u00f3k")
    except Exception:
        pass

    status_var = tk.StringVar(value="Adatok bet\u00f6lt\u00e9se...")
    status_lbl = ttk.Label(root, textvariable=status_var)
    status_lbl.grid(row=3, column=0, sticky="w", padx=6, pady=4)
    # Normalize initial status text to proper UTF-8
    try:
        status_var.set("Adatok bet\u00f6lt\u00e9se...")
    except Exception:
        pass

    # columns defined via columns_base below
    # Build columns with optional odds columns at the end; hide them by default via displaycolumns
    columns_base = ["D\u00e1tum", "Meccs", "Eredm\u00e9ny"] + list(MODEL_HEADERS.values())
    columns_all = columns_base + ["H odds", "D odds", "V odds"]
    tree = ttk.Treeview(root, columns=columns_all, show="headings")
    # Extend columns with odds-shopping column
    BEST_ODDS_COL = "Best odds"
    EV_COL = "EV"
    try:
        if BEST_ODDS_COL not in columns_all:
            columns_all = columns_base + [BEST_ODDS_COL, EV_COL, "H odds", "D odds", "V odds"]
            tree.configure(columns=columns_all)
    except Exception:
        # Safe fallback if reconfiguration fails
        pass

    # Enable click-to-sort on headers
    sort_by: int | None = None
    sort_reverse: bool = False

    def _on_sort(col_index: int) -> None:
        nonlocal sort_by, sort_reverse
        try:
            # Toggle sort order when same column clicked
            if sort_by == col_index:
                sort_reverse = not sort_reverse
            else:
                sort_by = col_index
                sort_reverse = False
            # Extract current values and sort
            items = tree.get_children("")

            def _key(i: str) -> Any:
                import re

                col = columns_all[col_index]
                s = str(tree.set(i, col)).strip()
                # Missing values go to the end on ascending
                if not s:
                    return (1, "")
                # Date column: ISO-like 'YYYY-MM-DD HH:MM' (optionally with 'UTC')
                if re.match(r"^\d{4}-\d{2}-\d{2}", s):
                    try:
                        parts = s.split()
                        dt_str = " ".join(parts[:2]) if len(parts) >= 2 else parts[0]
                        from datetime import datetime

                        dt = (
                            datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                            if " " in dt_str
                            else datetime.strptime(dt_str, "%Y-%m-%d")
                        )
                        return (0, dt)
                    except Exception:
                        pass
                # Percent at end e.g. '+5%'
                m = re.search(r"([+-]?\d+(?:[.,]\d+)?)%$", s)
                if m:
                    try:
                        return (0, float(m.group(1).replace(",", ".")))
                    except Exception:
                        pass
                # Probability label like '1 (65%)'
                m2 = re.search(r"\((\d+(?:\.\d+)?)%\)", s)
                if m2:
                    try:
                        return (0, float(m2.group(1)))
                    except Exception:
                        pass
                # Trailing numeric token (e.g., 'bookmaker 2.22' or '2.22')
                last = s.split()[-1]
                try:
                    return (0, float(last))
                except Exception:
                    return (0, s.lower())

            sorted_items = sorted(items, key=_key, reverse=sort_reverse)
            for idx, iid in enumerate(sorted_items):
                tree.move(iid, "", idx)
        except Exception:
            pass

    for idx, col in enumerate(columns_all):
        tree.heading(col, text=col, command=partial(_on_sort, idx))
        # Compact widths to avoid horizontal scrolling when odds columns are shown
        if col == "Meccs":
            width = 200
        elif col in ("D\u00e1tum",):
            width = 120
        elif col == EV_COL:
            width = 70
        elif col in ("H odds", "D odds", "V odds"):
            width = 80
        else:
            width = 100
        tree.column(col, width=width)
        tree.column(col, anchor=("w" if col in ("D\u00e1tum", "Meccs") else "center"))
        tree.column(col, stretch=False)
    # Fix first column header label to show proper accented text
    try:
        tree.heading(columns_all[0], text="D\u00e1tum")
    except Exception:
        pass
    try:
        tree.configure(displaycolumns=columns_base)
    except Exception:
        pass

    vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    # Right side stats panel placeholder
    stats_frame = ttk.LabelFrame(root, text="Modell statisztik\u00e1k")
    stats_frame.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=(12, 6), pady=(0, 6))
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(2, weight=0)

    # Row coloring tags for correctness (applied only in played view)
    try:
        tree.tag_configure("row_win", background="#e6ffe6")  # light green
        tree.tag_configure("row_lose", background="#ffeaea")  # light red
    except Exception:
        pass

    # Controls
    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=2, column=0, pady=8, sticky="w")

    # Normalize any incorrectly encoded texts on labels/buttons (best-effort)
    try:
        _normalize_widget_texts(root)
        _normalize_widget_texts(btn_frame)
    except Exception:
        pass

    # Default empty state until pipeline finishes: provision in-memory schema for safe refresh
    tmp_conn = sqlite3.connect(":memory:")
    try:
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            tmp_conn.executescript(f.read())
    except Exception:
        pass
    state = AppState(conn=tmp_conn, tree=tree)
    # Build stats label grid
    try:
        labels = [
            ("Minta", "count"),
            ("Tal\u00e1lati ar\u00e1ny", "hit"),
            ("\u00c1tlag odds", "avg_odds"),
            ("\u00c1tlag EV", "avg_ev"),
            ("Pozit\u00edv EV nyertesek", "pos_ev_win"),
        ]
        stats_vars: Dict[str, tk.StringVar] = {}
        for r, (lbl, key) in enumerate(labels):
            ttk.Label(stats_frame, text=lbl + ":").grid(row=r, column=0, sticky="w", padx=6, pady=4)
            var = tk.StringVar(value="-")
            stats_vars[key] = var
            ttk.Label(stats_frame, textvariable=var, anchor="e").grid(
                row=r, column=1, sticky="e", padx=6, pady=4
            )
        state.stats_frame = stats_frame
        state.stats_vars = stats_vars
        # Hide by default until a model is selected in played view
        try:
            stats_frame.grid_remove()
        except Exception:
            pass
    except Exception:
        pass

    # Bookmaker filter UI (combobox)
    ttk.Label(btn_frame, text="Fogad\u00f3iroda:").pack(side=tk.LEFT, padx=(0, 6))
    bm_var = tk.StringVar(value="(Mind)")
    bm_combo = ttk.Combobox(btn_frame, textvariable=bm_var, state="readonly", width=24)
    bm_combo.pack(side=tk.LEFT, padx=(0, 12))
    state.bm_combo_var = bm_var
    state.bm_combo = bm_combo

    # Model filter UI (combobox)
    ttk.Label(btn_frame, text="Modell:").pack(side=tk.LEFT, padx=(0, 6))
    model_var = tk.StringVar(value="(Mind)")
    model_combo = ttk.Combobox(btn_frame, textvariable=model_var, state="readonly", width=18)
    model_combo.pack(side=tk.LEFT, padx=(0, 12))
    state.model_combo_var = model_var
    state.model_combo = model_combo

    # Free-text search for team names
    ttk.Label(btn_frame, text="Keres\u00e9s:").pack(side=tk.LEFT, padx=(0, 6))
    search_var = tk.StringVar(value="")
    entry = ttk.Entry(btn_frame, textvariable=search_var, width=24)
    entry.pack(side=tk.LEFT, padx=(0, 12))
    state.search_var = search_var

    # Toggle button for played/upcoming view
    state.show_played = False

    def _update_played_btn_label() -> None:
        try:
            if state.show_played:
                played_btn.configure(text="K\u00f6zelg\u0151 m\u00e9rk\u0151z\u00e9sek")
            else:
                played_btn.configure(text="Befejezett m\u00e9rk\u0151z\u00e9sek")
        except Exception:
            pass

    def _on_toggle_played() -> None:
        try:
            state.show_played = not bool(getattr(state, "show_played", False))
            _apply_displaycolumns()
            _update_odds_headers()
            try:
                _update_winners_filter_visibility()
            except Exception:
                pass
            _update_played_btn_label()
            refresh_table(state)
        except Exception:
            pass

    played_btn = ttk.Button(
        btn_frame, text="Befejezett m\u00e9rk\u0151z\u00e9sek", command=_on_toggle_played
    )
    played_btn.pack(side=tk.LEFT, padx=(0, 12))
    _update_played_btn_label()

    # Winners-only checkbox (visible only for played view with a selected model)
    winners_var = tk.BooleanVar(value=False)
    state.winners_only = False  # dynamic attribute
    state.winners_only_var = winners_var

    def _on_winners_toggle() -> None:
        try:
            state.winners_only = bool(winners_var.get())
            refresh_table(state)
        except Exception:
            pass

    winners_chk = ttk.Checkbutton(
        btn_frame,
        text="Csak nyertes",
        variable=winners_var,
        command=_on_winners_toggle,
    )
    state.winners_only_chk = winners_chk

    def _update_winners_filter_visibility() -> None:
        try:
            has_model = state.selected_model_key is not None
            if getattr(state, "show_played", False) and has_model:
                # Show if not already mapped
                try:
                    if not winners_chk.winfo_ismapped():
                        winners_chk.pack(side=tk.LEFT, padx=(0, 12))
                except Exception:
                    winners_chk.pack(side=tk.LEFT, padx=(0, 12))
            else:
                # Hide and reset
                try:
                    winners_chk.pack_forget()
                except Exception:
                    pass
                try:
                    winners_var.set(False)
                    state.winners_only = False
                except Exception:
                    pass
        except Exception:
            pass

    def _on_search_change(*_args: Any) -> None:
        refresh_table(state)

    search_var.trace_add("write", _on_search_change)

    def _refresh_bm_list() -> None:
        try:
            cur = state.conn.execute("SELECT bookmaker_id, name FROM bookmakers ORDER BY name")
            names_by_id: Dict[int, str] = {int(r[0]): str(r[1]) for r in cur.fetchall()}
        except Exception:
            names_by_id = {}
        state.bm_names_by_id = names_by_id
        values = ["(Mind)"] + [
            names_by_id[k] for k in sorted(names_by_id, key=lambda x: names_by_id[x])
        ]
        bm_combo.configure(values=values)

    def _update_odds_headers() -> None:
        # Update odds column headers to show bookmaker name if selected
        try:
            if state.selected_bookmaker_id is not None and state.bm_names_by_id:
                name = state.bm_names_by_id.get(int(state.selected_bookmaker_id), "")
                tree.heading("H odds", text=f"H odds ({name})")
                tree.heading("D odds", text=f"D odds ({name})")
                tree.heading("V odds", text=f"V odds ({name})")
                # If a single model is selected, show bookmaker name in the tip-odds column header
                try:
                    if state.selected_model_key is not None:
                        tree.heading(BEST_ODDS_COL, text=name or "Tip odds")
                        tree.heading(EV_COL, text="EV")
                except Exception:
                    pass
            else:
                tree.heading("H odds", text="H odds")
                tree.heading("D odds", text="D odds")
                tree.heading("V odds", text="V odds")
                try:
                    if state.selected_model_key is not None:
                        tree.heading(BEST_ODDS_COL, text="Best odds")
                        # When no bookmaker is selected, show BEST EV header
                        tree.heading(EV_COL, text="BEST EV")
                except Exception:
                    pass
        except Exception:
            pass

    # Control which columns are visible based on selected model and bookmaker
    def _apply_displaycolumns() -> None:
        try:
            date_col = columns_base[0]
            match_col = columns_base[1]
            result_col = columns_base[2]
            display = [date_col, match_col]
            # Show real result column only in played-matches view
            if getattr(state, "show_played", False):
                display.append(result_col)
            has_model = state.selected_model_key is not None
            has_bookmaker = state.selected_bookmaker_id is not None
            if not has_model:
                display += list(MODEL_HEADERS.values())
            else:
                # mypy: selected_model_key is not None here, but not narrowed; cast to str
                model_key = cast(str, state.selected_model_key)
                disp = MODEL_HEADERS.get(model_key)
                if disp:
                    display.append(disp)
                else:
                    display += list(MODEL_HEADERS.values())
                # When a single model is selected, show the tip-odds column and EV
                if "BEST_ODDS_COL" in locals():
                    display.append(BEST_ODDS_COL)
                if "EV_COL" in locals():
                    display.append(EV_COL)
            # Detailed odds (H/D/V) only if bookmaker selected and NO single model selected
            if has_bookmaker and not has_model:
                display += ["H odds", "D odds", "V odds"]
            tree.configure(displaycolumns=display)
        except Exception:
            pass

    def _on_bm_selected(event: object) -> None:
        try:
            sel = bm_var.get()
            if sel == "(Mind)":
                state.selected_bookmaker_id = None
            else:
                if state.bm_names_by_id:
                    # reverse map name->id
                    rev = {v: k for k, v in state.bm_names_by_id.items()}
                    state.selected_bookmaker_id = rev.get(sel)
            _apply_displaycolumns()
            _update_odds_headers()
            refresh_table(state)
        except Exception:
            pass

    bm_combo.bind("<<ComboboxSelected>>", _on_bm_selected)

    # Populate model list and handle selection
    def _refresh_model_list() -> None:
        try:
            names_disp = list(MODEL_HEADERS.values())
            model_combo.configure(values=["(Mind)"] + names_disp)
        except Exception:
            pass

    def _on_model_selected(event: object) -> None:
        try:
            sel = model_var.get()
            if sel == "(Mind)":
                state.selected_model_key = None
            else:
                rev = {v: k for k, v in MODEL_HEADERS.items()}
                state.selected_model_key = rev.get(sel)
            _apply_displaycolumns()
            try:
                _update_winners_filter_visibility()
            except Exception:
                pass
            refresh_table(state)
        except Exception:
            pass

    model_combo.bind("<<ComboboxSelected>>", _on_model_selected)

    ttk.Button(
        btn_frame,
        text="Friss\u00edt\u00e9s",
        command=lambda: refresh_table(state),
    ).pack(side=tk.LEFT)

    # Double-click to show odds for the selected match from DB
    def _on_double_click(event: object) -> None:
        try:
            sel = tree.focus()
            if not sel:
                return
            fx_id = int(sel)
            _show_odds_window_db(state.conn, tree, fx_id)
        except Exception:
            pass

    tree.bind("<Double-1>", _on_double_click)

    # Run pipeline; keep window even on error
    try:
        write_conn = _ensure_db()
        # Always refresh holnapi meccsek, and per-fixture csak akkor mentünk,
        # ha minden modell tudott prediktálni.
        fixtures = _select_fixtures_for_tomorrow()
        for fx in fixtures:
            try:
                _predict_then_persist_if_complete(write_conn, fx)
            except Exception:
                pass

        try:
            write_conn.close()
        except Exception:
            pass
        # Switch to read-only connection for the GUI
        ro = _open_ro()
        state.conn = ro
        # After switching to DB, populate bookmaker list
        try:
            _refresh_bm_list()
        except Exception:
            pass
        _refresh_model_list()
        _apply_displaycolumns()
        _update_odds_headers()
        try:
            status_var.set("K\u00e9sz.")
        except Exception:
            pass
        status_var.set("K\u00e9sz.")
    except Exception as exc:
        print(f"[GUI] Hiba a feldolgoz\u00e1s sor\u00e1n: {exc}")
        try:
            messagebox.showerror("Hiba", f"Hiba a feldolgoz\u00e1s sor\u00e1n: {exc}")
        except Exception:
            pass
        # If DB exists, try opening read-only to show existing rows
        try:
            if os.path.exists(DB_PATH):
                state.conn = _open_ro()
                try:
                    _refresh_bm_list()
                except Exception:
                    pass
                try:
                    _refresh_model_list()
                    _apply_displaycolumns()
                except Exception:
                    pass
        except Exception:
            pass
        status_var.set("Hiba t\u00f6rt\u00e9nt.")

    refresh_table(state)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_app()
