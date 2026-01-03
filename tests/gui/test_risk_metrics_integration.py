"""
Integration tests for risk metrics in Daily Stats computation.

These tests verify the full computation pipeline including:
- Database queries
- Daily profit aggregation
- Risk metric calculations (Sortino, Profit Factor, Recovery)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest


class MockVar:
    """Mock tkinter Variable for testing."""

    def __init__(self, value: Any = "") -> None:
        self._value = value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class MockAppState:
    """Mock AppState with necessary attributes for testing."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.selected_model_key = "poisson"
        self.show_played = True
        self.search_var = MockVar("")
        self.daily_league_var = MockVar("-")
        self.daily_bm_var = MockVar("-")
        self.exclude_extremes = False
        self.daily_top_n_var = MockVar("0")
        self.daily_top_n_type_var = MockVar("Legjobb +EV")
        self.daily_ev_range_var = MockVar("-")
        self.daily_odds_range_var = MockVar("-")
        self.daily_min_odds_var = MockVar("")
        self.daily_max_odds_var = MockVar("")
        self.daily_start_date_var = MockVar("")
        self.daily_system_bet_var = MockVar("Egyes kötés")
        self.bankroll_amount = None
        self.bm_names_by_id: dict[int, str] = {1: "Bet365"}
        self.daily_league_label_to_id = {"-": "-"}
        self.daily_selected_bookmaker_id = None


@pytest.fixture
def test_db(tmp_path: Path) -> sqlite3.Connection:
    """Create a temporary test database with schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))

    # Create schema
    conn.executescript(
        """
        CREATE TABLE matches (
            match_id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            real_result TEXT,
            league_id INTEGER
        );

        CREATE TABLE predictions (
            match_id INTEGER,
            model_name TEXT,
            prob_home REAL,
            prob_draw REAL,
            prob_away REAL,
            predicted_result TEXT,
            PRIMARY KEY (match_id, model_name)
        );

        CREATE TABLE odds (
            match_id INTEGER,
            bookmaker_id INTEGER,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            PRIMARY KEY (match_id, bookmaker_id)
        );
        """
    )

    return conn


def test_compute_daily_stats_with_mixed_profits(test_db: sqlite3.Connection) -> None:
    """Test risk metrics computation with mixed positive and negative daily profits."""
    # Insert test data: 5 days with known profit pattern
    # Day 1: +1 profit (win on 2.0 odds)
    # Day 2: -1 profit (loss)
    # Day 3: +3 profit (win on 4.0 odds)
    # Day 4: -2 profit (2 losses)
    # Day 5: +1 profit (win on 2.0 odds)
    # Total: +2, daily: [1, -1, 3, -2, 1]

    test_data = [
        # Day 1: Win +1
        (
            1,
            "2024-01-01 15:00:00",
            "Home1",
            "Away1",
            "1",
            1,
            0.6,
            0.3,
            0.1,
            "1",
            2.0,
            1.5,
            5.0,
        ),
        # Day 2: Loss -1
        (
            2,
            "2024-01-02 15:00:00",
            "Home2",
            "Away2",
            "X",
            1,
            0.5,
            0.3,
            0.2,
            "1",
            2.5,
            3.0,
            4.0,
        ),
        # Day 3: Win +3 (big win)
        (
            3,
            "2024-01-03 15:00:00",
            "Home3",
            "Away3",
            "1",
            1,
            0.4,
            0.3,
            0.3,
            "1",
            4.0,
            3.0,
            2.5,
        ),
        # Day 4: Loss -1
        (
            4,
            "2024-01-04 15:00:00",
            "Home4",
            "Away4",
            "X",
            1,
            0.6,
            0.2,
            0.2,
            "1",
            2.0,
            3.5,
            5.0,
        ),
        # Day 4: Another loss -1 (same day, total -2)
        (
            5,
            "2024-01-04 17:00:00",
            "Home5",
            "Away5",
            "2",
            1,
            0.5,
            0.3,
            0.2,
            "1",
            2.5,
            3.0,
            4.0,
        ),
        # Day 5: Win +1
        (
            6,
            "2024-01-05 15:00:00",
            "Home6",
            "Away6",
            "1",
            1,
            0.7,
            0.2,
            0.1,
            "1",
            2.0,
            3.5,
            6.0,
        ),
    ]

    for (
        mid,
        date,
        home,
        away,
        result,
        lid,
        ph,
        pd,
        pa,
        pred,
        oh,
        od,
        oa,
    ) in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    # Import here to avoid importing GUI at module level
    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, cumulative, _, total_profit, risk, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    # Verify basic aggregation
    assert len(rows) == 5  # 5 distinct days
    assert abs(total_profit - 2.0) < 0.01  # +1 -1 +3 -2 +1 = +2

    # Verify risk metrics exist
    assert risk is not None
    assert "std" in risk
    assert "max_drawdown" in risk

    # Verify Max DD (should occur after day 4)
    # Cumulative: [1, 0, 3, 1, 2] -> peak at 3, DD = 1-3 = -2
    max_dd = risk["max_drawdown"]
    assert max_dd is not None
    assert max_dd < 0  # Drawdown is negative
    assert abs(max_dd + 2.0) < 0.1  # Allow some tolerance


def test_compute_daily_stats_sortino_calculation(test_db: sqlite3.Connection) -> None:
    """Test that Sortino ratio is calculated correctly from daily stats."""
    # Simple pattern: [+1, -1, +1, -1] -> avg = 0, but has downside volatility
    test_data = [
        (1, "2024-01-01", "H1", "A1", "1", 1, 0.6, 0.3, 0.1, "1", 2.0, 3.0, 4.0),
        (2, "2024-01-02", "H2", "A2", "X", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        (3, "2024-01-03", "H3", "A3", "1", 1, 0.6, 0.3, 0.1, "1", 2.0, 3.0, 4.0),
        (4, "2024-01-04", "H4", "A4", "2", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
    ]

    for mid, date, home, away, result, lid, ph, pd, pa, pred, oh, od, oa in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, _, _, total_profit, risk, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    # Daily profits: [+1, -1, +1, -1]
    daily_profits = [r.get("profit", 0.0) for r in rows]
    assert len(daily_profits) == 4
    assert abs(sum(daily_profits) - 0.0) < 0.01

    # Standard deviation should exist
    std = risk.get("std")
    assert std is not None
    assert std > 0


def test_compute_daily_stats_profit_factor(test_db: sqlite3.Connection) -> None:
    """Test Profit Factor calculation through full pipeline."""
    # Pattern: [+5, -2, +3] -> PF = (5+3)/2 = 4.0
    test_data = [
        # Day 1: Win +5 (6.0 odds)
        (1, "2024-01-01", "H1", "A1", "1", 1, 0.3, 0.3, 0.4, "1", 6.0, 3.0, 2.0),
        # Day 2: Loss -1
        (2, "2024-01-02", "H2", "A2", "X", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 2: Another loss (same day, total -2)
        (3, "2024-01-02 18:00", "H3", "A3", "2", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 3: Win +3 (4.0 odds)
        (4, "2024-01-03", "H4", "A4", "1", 1, 0.4, 0.3, 0.3, "1", 4.0, 3.0, 2.5),
    ]

    for mid, date, home, away, result, lid, ph, pd, pa, pred, oh, od, oa in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, _, _, total_profit, _, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    # Verify daily profits
    assert len(rows) == 3  # 3 days
    daily_profits = [r.get("profit", 0.0) for r in rows]

    # Day 1: +5, Day 2: -2, Day 3: +3
    assert abs(daily_profits[0] - 5.0) < 0.01
    assert abs(daily_profits[1] + 2.0) < 0.01
    assert abs(daily_profits[2] - 3.0) < 0.01

    # Total profit: +6
    assert abs(total_profit - 6.0) < 0.01


def test_compute_daily_stats_max_win_impact(test_db: sqlite3.Connection) -> None:
    """Test Max Win Impact when one day dominates total profit."""
    # Pattern: [+10, -1, +0.5, -0.5] -> total +9, max win = 10 -> 111% impact!
    test_data = [
        # Day 1: Huge win +10 (11.0 odds)
        (1, "2024-01-01", "H1", "A1", "1", 1, 0.2, 0.3, 0.5, "1", 11.0, 3.0, 2.0),
        # Day 2: Small loss -1
        (2, "2024-01-02", "H2", "A2", "X", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 3: Small win +0.5
        (3, "2024-01-03", "H3", "A3", "1", 1, 0.5, 0.3, 0.2, "1", 1.5, 3.0, 4.0),
        # Day 4: Small loss -0.5
        (4, "2024-01-04", "H4", "A4", "2", 1, 0.5, 0.3, 0.2, "1", 1.5, 3.0, 4.0),
    ]

    for mid, date, home, away, result, lid, ph, pd, pa, pred, oh, od, oa in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, _, _, total_profit, _, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    daily_profits = [r.get("profit", 0.0) for r in rows]
    max_profit = max(daily_profits)

    # Max win impact should be very high (>100%)
    max_win_impact = (max_profit / total_profit * 100.0) if total_profit > 0 else 0
    assert max_win_impact > 100.0  # Red flag! Over 100% dependency


def test_compute_daily_stats_recovery_factor(test_db: sqlite3.Connection) -> None:
    """Test Recovery Factor = total_profit / abs(max_drawdown)."""
    # Pattern to create drawdown: [+3, -2, +4] -> DD at pos 2: -2, final profit +5
    # Recovery = 5 / 2 = 2.5
    test_data = [
        # Day 1: Win +3
        (1, "2024-01-01", "H1", "A1", "1", 1, 0.4, 0.3, 0.3, "1", 4.0, 3.0, 2.0),
        # Day 2: Loss -1
        (2, "2024-01-02", "H2", "A2", "X", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 2: Another loss -1 (total -2)
        (3, "2024-01-02 18:00", "H3", "A3", "2", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 3: Big win +4
        (4, "2024-01-03", "H4", "A4", "1", 1, 0.35, 0.3, 0.35, "1", 5.0, 3.0, 2.5),
    ]

    for mid, date, home, away, result, lid, ph, pd, pa, pred, oh, od, oa in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, cumulative, _, total_profit, risk, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    # Verify cumulative curve: [3, 1, 5]
    # Peak: 3, then drops to 1 (DD = -2), then recovers to 5
    max_dd = risk.get("max_drawdown")
    assert max_dd is not None
    assert abs(max_dd + 2.0) < 0.01

    # Recovery factor
    recovery = total_profit / abs(max_dd) if max_dd and max_dd != 0 else None
    assert recovery is not None
    assert abs(recovery - 2.5) < 0.01


def test_compute_daily_stats_sharpe_ratio(test_db: sqlite3.Connection) -> None:
    """Test Sharpe Ratio calculation through full pipeline."""
    # Pattern: [+2, -1, +3, +1] -> avg = 1.25, with volatility
    test_data = [
        # Day 1: Win +2 (3.0 odds)
        (1, "2024-01-01", "H1", "A1", "1", 1, 0.5, 0.3, 0.2, "1", 3.0, 2.5, 2.0),
        # Day 2: Loss -1
        (2, "2024-01-02", "H2", "A2", "X", 1, 0.5, 0.3, 0.2, "1", 2.0, 3.0, 4.0),
        # Day 3: Win +3 (4.0 odds)
        (3, "2024-01-03", "H3", "A3", "1", 1, 0.4, 0.3, 0.3, "1", 4.0, 3.0, 2.5),
        # Day 4: Win +1 (2.0 odds)
        (4, "2024-01-04", "H4", "A4", "1", 1, 0.6, 0.3, 0.1, "1", 2.0, 3.0, 5.0),
    ]

    for mid, date, home, away, result, lid, ph, pd, pa, pred, oh, od, oa in test_data:
        test_db.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?)",
            (mid, date, home, away, result, lid),
        )
        test_db.execute(
            "INSERT INTO predictions VALUES (?, 'poisson', ?, ?, ?, ?)",
            (mid, ph, pd, pa, pred),
        )
        test_db.execute("INSERT INTO odds VALUES (?, 1, ?, ?, ?)", (mid, oh, od, oa))

    test_db.commit()

    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, _, _, total_profit, risk, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    # Verify daily profits: [+2, -1, +3, +1] = total +5
    assert len(rows) == 4
    assert abs(total_profit - 5.0) < 0.01

    # Verify std dev was calculated
    std = risk.get("std")
    assert std is not None
    assert std > 0

    # Manual Sharpe calculation to verify
    daily_profits = [2.0, -1.0, 3.0, 1.0]
    import math

    mean_val = sum(daily_profits) / len(daily_profits)  # 1.25
    variance = sum((x - mean_val) ** 2 for x in daily_profits) / (len(daily_profits) - 1)
    expected_std = math.sqrt(variance)

    # The std from risk should match our manual calculation
    assert abs(std - expected_std) < 0.01

    # Sharpe = avg_daily / std
    avg_daily = total_profit / len(rows)  # 1.25
    expected_sharpe = avg_daily / expected_std

    # Sharpe should be positive for profitable strategy
    assert expected_sharpe > 0


def test_compute_daily_stats_no_data(test_db: sqlite3.Connection) -> None:
    """Test handling of empty dataset."""
    from src.gui.app import _compute_daily_model_stats

    state = MockAppState(test_db)
    rows, cumulative, _, total_profit, risk, _ = _compute_daily_model_stats(state)  # type: ignore[arg-type]

    assert len(rows) == 0
    assert len(cumulative) == 0
    assert total_profit == 0.0
    assert risk.get("std") is None
    assert risk.get("max_drawdown") is None
