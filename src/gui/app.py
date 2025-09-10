from __future__ import annotations

import os
import sqlite3
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, Iterable, List, Mapping, Tuple, cast

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
    "Holnapi meccsek �s predikci�k": "Holnapi meccsek és predikciók",
    "Holnapi meccsek �cs predikciƈk": "Holnapi meccsek és predikciók",
    "Adatok bet�\u0014lt�cse...": "Adatok betöltése...",
    "Fogadƈiroda:": "Fogadóiroda:",
    "Keres�cs:": "Keresés:",
    "Friss��t�cs": "Frissítés",
    "D�tum": "Dátum",
    "K�csz.": "Kész.",
    "Hiba t�\u0014rt�cnt.": "Hiba történt.",
}


def _fix_mojibake(s: str) -> str:
    try:
        out = s
        for bad, good in _MOJIBAKE_MAP.items():
            if bad in out:
                out = out.replace(bad, good)
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
        values = [disp_dt, f"{home} - {away}"]
        # If a bookmaker is selected, pull odds once per match to render beneath the probabilities
        match_odds: Tuple[float, float, float] | None = None
        if state.selected_bookmaker_id is not None:
            try:
                cur = state.conn.execute(
                    "SELECT odds_home, odds_draw, odds_away FROM odds WHERE match_id = ? AND bookmaker_id = ? LIMIT 1",
                    (int(mid), int(state.selected_bookmaker_id)),
                )
                r = cur.fetchone()
                if r is not None:
                    match_odds = (float(r[0]), float(r[1]), float(r[2]))
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
        # Append odds columns at the end if selected; otherwise leave empty cells
        if match_odds is not None and state.selected_bookmaker_id is not None:
            oh, od, oa = match_odds
            values.extend([f"{oh:.2f}", f"{od:.2f}", f"{oa:.2f}"])
        else:
            values.extend(["", "", ""])
        # use match_id as iid for easy retrieval on double-click
        # Zebra striping for readability
        tag = "odd" if (len(state.tree.get_children()) % 2 == 1) else "even"
        state.tree.insert("", "end", iid=str(int(mid)), values=values, tags=(tag,))


def _show_odds_window(tree: ttk.Treeview, fixture_id: int) -> None:
    try:
        svc = OddsService()
        odds_list = svc.get_fixture_odds(fixture_id)
        names = svc.get_fixture_bookmakers(fixture_id)
    except Exception as exc:
        messagebox.showerror("Hiba", f"Odds lekĂ©rĂ©s sikertelen: {exc}")
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds â€“ {fixture_id}")
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
        messagebox.showerror("Hiba", f"Odds betĂ¶ltĂ©se az adatbĂˇzisbĂłl sikertelen: {exc}")
        return

    top = tk.Toplevel(tree.winfo_toplevel())
    top.title(f"Odds â€“ {fixture_id}")
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
    root.title("Holnapi meccsek Ă©s predikciĂłk")
    root.geometry("1200x600")
    # Ensure proper accented title regardless of source encoding
    try:
        root.title("Holnapi meccsek és predikciók")
    except Exception:
        pass

    status_var = tk.StringVar(value="Adatok betĂ¶ltĂ©se...")
    status_lbl = ttk.Label(root, textvariable=status_var)
    status_lbl.grid(row=3, column=0, sticky="w", padx=6, pady=4)
    # Normalize initial status text to proper UTF-8
    try:
        status_var.set("Adatok betöltése...")
    except Exception:
        pass

    # columns defined via columns_base below
    # Build columns with optional odds columns at the end; hide them by default via displaycolumns
    columns_base = ["DĂˇtum", "Meccs"] + list(MODEL_HEADERS.values())
    columns_all = columns_base + ["H odds", "D odds", "V odds"]
    tree = ttk.Treeview(root, columns=columns_all, show="headings")

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
                val = tree.set(i, columns_all[col_index])
                try:
                    return float(val)
                except Exception:
                    return str(val)

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
        elif col in ("Dátum", "D��tum"):
            width = 120
        elif col in ("H odds", "D odds", "V odds"):
            width = 80
        else:
            width = 100
        tree.column(col, width=width)
        tree.column(col, anchor=("w" if col in ("Dátum", "D��tum", "Meccs") else "center"))
        tree.column(col, stretch=False)
    # Fix first column header label to show proper accented text
    try:
        tree.heading(columns_all[0], text="Dátum")
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
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    # Controls
    btn_frame = ttk.Frame(root)
    btn_frame.grid(row=2, column=0, pady=8, sticky="w")

    # Normalize any incorrectly encoded texts on labels/buttons (best-effort)
    try:
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

    # Bookmaker filter UI (combobox)
    ttk.Label(btn_frame, text="FogadĂłiroda:").pack(side=tk.LEFT, padx=(0, 6))
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
    ttk.Label(btn_frame, text="KeresĂ©s:").pack(side=tk.LEFT, padx=(0, 6))
    search_var = tk.StringVar(value="")
    entry = ttk.Entry(btn_frame, textvariable=search_var, width=24)
    entry.pack(side=tk.LEFT, padx=(0, 12))
    state.search_var = search_var

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
            else:
                tree.heading("H odds", text="H odds")
                tree.heading("D odds", text="D odds")
                tree.heading("V odds", text="V odds")
        except Exception:
            pass

    # Control which columns are visible based on selected model and bookmaker
    def _apply_displaycolumns() -> None:
        try:
            date_col = columns_base[0]
            match_col = columns_base[1]
            display = [date_col, match_col]
            if state.selected_model_key is None:
                display += list(MODEL_HEADERS.values())
            else:
                disp = MODEL_HEADERS.get(state.selected_model_key)
                if disp:
                    display.append(disp)
                else:
                    display += list(MODEL_HEADERS.values())
            if state.selected_bookmaker_id is not None:
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
            refresh_table(state)
        except Exception:
            pass

    model_combo.bind("<<ComboboxSelected>>", _on_model_selected)

    ttk.Button(
        btn_frame,
        text="Friss\u00EDt\u00E9s",
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
        # After switching to DB, populate bookmaker list
        try:
            _refresh_bm_list()
        except Exception:
            pass
        _refresh_model_list()
        _apply_displaycolumns()
        _update_odds_headers()
        try:
            status_var.set("Kész.")
        except Exception:
            pass
        status_var.set("KĂ©sz.")
    except Exception as exc:
        print(f"[GUI] Hiba a feldolgozĂˇs sorĂˇn: {exc}")
        try:
            messagebox.showerror("Hiba", f"Hiba a feldolgozĂˇs sorĂˇn: {exc}")
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
        status_var.set("Hiba tĂ¶rtĂ©nt.")

    refresh_table(state)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_app()
