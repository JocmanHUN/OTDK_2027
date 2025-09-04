from __future__ import annotations

import os
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, Iterable, List, Mapping, Tuple

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


def _insert_matches(conn: sqlite3.Connection, fixtures: Iterable[Mapping[str, Any]]) -> None:
    mrepo = MatchesRepoSqlite(conn)
    for r in fixtures:
        fx_id = int(r["fixture_id"])
        league_id = int(r["league_id"])
        season = int(r["season"]) if r.get("season") is not None else datetime.now().year
        when = r.get("date_utc")
        if not isinstance(when, datetime):
            when = datetime.now(timezone.utc)
        home_name = r.get("home_name") or str(r.get("home_id") or "?")
        away_name = r.get("away_name") or str(r.get("away_id") or "?")

        _ensure_league(conn, league_id)

        existing = mrepo.get_by_id(fx_id)
        if existing is not None:
            continue
        m = DbMatch(
            id=fx_id,
            league_id=league_id,
            season=season,
            date=when,
            home_team=str(home_name),
            away_team=str(away_name),
            real_result=None,
        )
        mrepo.insert(m)


def _persist_odds(conn: sqlite3.Connection, fixtures: Iterable[Mapping[str, Any]]) -> None:
    """Fetch odds via API and persist them into DB for each fixture id.

    - Ensures bookmaker names exist in `bookmakers` table (FK dependency).
    - Upserts odds by (match_id, bookmaker_id).
    """
    svc = OddsService()
    orepo = OddsRepoSqlite(conn)
    brepo = BookmakersRepoSqlite(conn)

    for r in fixtures:
        fx_id = r.get("fixture_id")
        if not isinstance(fx_id, int):
            continue
        try:
            odds_list = svc.get_fixture_odds(int(fx_id))
            bm_names = svc.get_fixture_bookmakers(int(fx_id))
        except Exception:
            continue

        # Ensure bookmaker names
        for bid, name in bm_names.items():
            try:
                if brepo.get_by_id(int(bid)) is None:
                    brepo.insert(Bookmaker(id=int(bid), name=str(name)))
            except Exception:
                # ignore races/duplicates
                pass

        # Upsert odds
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


def _predict_and_store(conn: sqlite3.Connection, fixtures: Iterable[Mapping[str, Any]]) -> None:
    history = HistoryService()
    ctx_builder = ContextBuilder(history=history)
    models = default_models()
    agg = PredictionAggregatorImpl()
    prepo = PredictionsRepoSqlite(conn)

    for r in fixtures:
        try:
            fx_id_val = r.get("fixture_id")
            league_id_val = r.get("league_id")
            if not isinstance(fx_id_val, int) or not isinstance(league_id_val, int):
                continue
            fx_id = int(fx_id_val)
            league_id = int(league_id_val)
            season = int(r["season"]) if r.get("season") is not None else datetime.now().year
            home_id = r.get("home_id")
            away_id = r.get("away_id")
            if not isinstance(home_id, int) or not isinstance(away_id, int):
                continue

            ctx = ctx_builder.build_from_meta(
                fixture_id=fx_id,
                league_id=league_id,
                season=season,
                home_team_id=home_id,
                away_team_id=away_id,
            )

            match = DomainMatch(
                fixture_id=FixtureId(fx_id),
                league_id=LeagueId(league_id),
                season=season,
                kickoff_utc=datetime.now(timezone.utc),
                home_name=str(r.get("home_name") or home_id),
                away_name=str(r.get("away_name") or away_id),
                status=MatchStatus.SCHEDULED,
            )

            preds = agg.run_all(models, match, ctx)
            for p in preds:
                probs = p.probs
                if probs is None:
                    continue
                pr = probs
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
        except Exception:
            continue


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

    matches, preds_by_match = _read_tomorrow_from_db(state.conn)

    for mid, dt_s, home, away in matches:
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
        values = [disp_dt, f"{home} - {away}"]
        for key in MODEL_HEADERS.keys():
            trip = model_map.get(key)
            if trip:
                ph, pd, pa, pref = trip
                lbl, pct = _best_label_and_pct(ph, pd, pa, pref)
                values.append(f"{lbl} ({pct:.0f}%)")
            else:
                values.append("-")
        # use match_id as iid for easy retrieval on double-click
        state.tree.insert("", "end", iid=str(int(mid)), values=values)


def _show_odds_window(tree: ttk.Treeview, fixture_id: int) -> None:
    try:
        svc = OddsService()
        odds_list = svc.get_fixture_odds(fixture_id)
        names = svc.get_fixture_bookmakers(fixture_id)
    except Exception as exc:
        messagebox.showerror("Hiba", f"Odds lekérés sikertelen: {exc}")
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds – {fixture_id}")
    top.geometry("600x400")
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
        messagebox.showerror("Hiba", f"Odds betöltése az adatbázisból sikertelen: {exc}")
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds – {fixture_id}")
    top.geometry("600x400")
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

    for o in odds_list:
        bid = int(o.bookmaker_id)
        bm_name = names.get(bid, str(bid))
        home = f"{float(o.home):.2f}"
        draw = f"{float(o.draw):.2f}"
        away = f"{float(o.away):.2f}"
        tv.insert("", "end", values=[bm_name, home, draw, away])


def _tomorrow_bounds_iso() -> Tuple[str, str]:
    today_utc = datetime.now(timezone.utc).date()
    start = datetime.combine(
        today_utc + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _get_tomorrow_match_ids(conn: sqlite3.Connection) -> List[int]:
    start, end = _tomorrow_bounds_iso()
    cur = conn.execute(
        "SELECT match_id FROM matches WHERE date >= ? AND date < ? ORDER BY date",
        (start, end),
    )
    return [int(r[0]) for r in cur.fetchall()]


def _models_for_match(conn: sqlite3.Connection, match_id: int) -> set[str]:
    cur = conn.execute(
        "SELECT DISTINCT model_name FROM predictions WHERE match_id = ?",
        (match_id,),
    )
    return {str(r[0]) for r in cur.fetchall()}


def _default_model_names() -> List[str]:
    names: List[str] = []
    for m in default_models():
        try:
            names.append(str(m.name.value))
        except Exception:
            names.append(str(m.name))
    return names


def run_app() -> None:
    # Build GUI first so errors can be shown reliably
    root = tk.Tk()
    root.title("Holnapi meccsek és predikciók")
    root.geometry("1200x600")

    status_var = tk.StringVar(value="Adatok betöltése...")
    status_lbl = ttk.Label(root, textvariable=status_var)
    status_lbl.grid(row=3, column=0, sticky="w", padx=6, pady=4)

    columns = ["Dátum", "Meccs"] + list(MODEL_HEADERS.values())
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=130 if col != "Meccs" else 260, anchor=tk.W)

    vsb = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(root, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Controls
    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=2, column=0, pady=8, sticky="w")

    # Default empty state until pipeline finishes: provision in-memory schema for safe refresh
    tmp_conn = sqlite3.connect(":memory:")
    try:
        with open(MIGRATION_FILE, "r", encoding="utf-8") as f:
            tmp_conn.executescript(f.read())
    except Exception:
        pass
    state = AppState(conn=tmp_conn, tree=tree)
    ttk.Button(btn_frame, text="Frissítés", command=lambda: refresh_table(state)).pack(side=tk.LEFT)

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
        model_names = _default_model_names()

        mids = _get_tomorrow_match_ids(write_conn)
        full = False
        if mids:
            full = all(
                set(model_names).issubset(_models_for_match(write_conn, mid)) for mid in mids
            )

        if not full:
            fixtures = _select_fixtures_for_tomorrow()
            _insert_matches(write_conn, fixtures)
            # Persist odds so the GUI odds window can read from DB
            _persist_odds(write_conn, fixtures)
            # Compute only for fixtures missing some models
            todo: List[Mapping[str, Any]] = []
            for r in fixtures:
                fx_id = r.get("fixture_id")
                if isinstance(fx_id, int):
                    present = _models_for_match(write_conn, int(fx_id))
                    if not set(model_names).issubset(present):
                        todo.append(r)
            if todo:
                _predict_and_store(write_conn, todo)

        try:
            write_conn.close()
        except Exception:
            pass
        # Switch to read-only connection for the GUI
        ro = _open_ro()
        state.conn = ro
        status_var.set("Kész.")
    except Exception as exc:
        print(f"[GUI] Hiba a feldolgozás során: {exc}")
        try:
            messagebox.showerror("Hiba", f"Hiba a feldolgozás során: {exc}")
        except Exception:
            pass
        # If DB exists, try opening read-only to show existing rows
        try:
            if os.path.exists(DB_PATH):
                state.conn = _open_ro()
        except Exception:
            pass
        status_var.set("Hiba történt.")

    refresh_table(state)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_app()
