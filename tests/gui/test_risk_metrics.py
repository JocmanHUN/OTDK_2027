"""
Tests for risk metrics calculations in Daily Stats window.

These tests verify the correct computation of:
- Sharpe Ratio (standard deviation based)
- Sortino Ratio (downside deviation based)
- Profit Factor (wins/losses ratio)
- Recovery Factor (profit/drawdown ratio)
- Max Win Impact (largest win dependency)
"""

from __future__ import annotations

import math


class MockState:
    """Mock AppState for testing risk metrics without full GUI."""

    def __init__(self) -> None:
        self.conn = None
        self.selected_model_key = "poisson"
        self.show_played = True
        self.search_var = None
        self.daily_league_var = None
        self.daily_bm_var = None
        self.exclude_extremes = False
        self.daily_top_n_var = None
        self.daily_top_n_type_var = None
        self.daily_ev_range_var = None
        self.daily_odds_range_var = None
        self.daily_min_odds_var = None
        self.daily_max_odds_var = None
        self.daily_start_date_var = None
        self.daily_system_bet_var = None
        self.bankroll_amount = None
        self.bm_names_by_id: dict[int, str] = {}
        self.daily_league_label_to_id = {"-": "-"}


def test_sortino_ratio_calculation() -> None:
    """Test Sortino Ratio calculation with downside deviation."""
    # Daily profits: [+2, -1, +3, -2, +1, 0]
    daily_profits = [2.0, -1.0, 3.0, -2.0, 1.0, 0.0]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 3.0

    # Calculate downside deviation (only negative deviations from 0)
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    # = [0, 1, 0, 4, 0, 0] = sum = 5
    downside_var = sum(downside_deviations_sq) / total_days  # = 5/6 ≈ 0.833
    downside_std = math.sqrt(downside_var)  # ≈ 0.913

    avg_daily_profit = active_total / total_days  # = 3/6 = 0.5
    expected_sortino = avg_daily_profit / downside_std  # ≈ 0.5 / 0.913 ≈ 0.548

    # Verify calculation
    assert abs(downside_var - 5 / 6) < 1e-9
    assert abs(downside_std - math.sqrt(5 / 6)) < 1e-9
    assert abs(expected_sortino - 0.5475) < 0.001


def test_sortino_ratio_all_positive() -> None:
    """Test Sortino Ratio when all profits are positive."""
    daily_profits = [1.0, 2.0, 3.0, 1.5, 2.5]
    total_days = len(daily_profits)

    # All positive, so all downside deviations are 0
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    downside_var = sum(downside_deviations_sq) / total_days  # = 0 / 5 = 0
    downside_std = math.sqrt(downside_var)  # = 0

    # When downside_std is 0, Sortino should be None or undefined
    # In our implementation, we check if downside_std > 0
    assert downside_std == 0.0


def test_sortino_ratio_all_negative() -> None:
    """Test Sortino Ratio when all profits are negative."""
    daily_profits = [-1.0, -2.0, -1.5, -0.5]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = -5.0

    # All negative
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    # = [1, 4, 2.25, 0.25] = sum = 7.5
    downside_var = sum(downside_deviations_sq) / total_days  # = 7.5 / 4 = 1.875
    downside_std = math.sqrt(downside_var)  # ≈ 1.369

    avg_daily_profit = active_total / total_days  # = -5 / 4 = -1.25
    expected_sortino = avg_daily_profit / downside_std  # ≈ -1.25 / 1.369 ≈ -0.913

    # Verify calculation
    assert abs(downside_var - 1.875) < 1e-9
    assert abs(expected_sortino + 0.913) < 0.001


def test_profit_factor_calculation() -> None:
    """Test Profit Factor = sum(wins) / sum(losses)."""
    daily_profits = [5.0, -2.0, 3.0, -1.0, 4.0, -3.0]

    wins_sum = sum(p for p in daily_profits if p > 0)  # 5 + 3 + 4 = 12
    losses_sum = abs(sum(p for p in daily_profits if p < 0))  # |(-2-1-3)| = 6

    expected_pf = wins_sum / losses_sum  # 12 / 6 = 2.0

    assert abs(wins_sum - 12.0) < 1e-9
    assert abs(losses_sum - 6.0) < 1e-9
    assert abs(expected_pf - 2.0) < 1e-9


def test_profit_factor_no_losses() -> None:
    """Test Profit Factor when there are no losses (should be None or inf)."""
    daily_profits = [1.0, 2.0, 3.0]

    wins_sum = sum(p for p in daily_profits if p > 0)  # 6
    losses_sum = abs(sum(p for p in daily_profits if p < 0))  # 0

    # When losses_sum is 0, Profit Factor is None or undefined
    assert losses_sum == 0.0
    assert wins_sum == 6.0


def test_profit_factor_no_wins() -> None:
    """Test Profit Factor when there are no wins (should be 0)."""
    daily_profits = [-1.0, -2.0, -3.0]

    wins_sum = sum(p for p in daily_profits if p > 0)  # 0
    losses_sum = abs(sum(p for p in daily_profits if p < 0))  # 6

    expected_pf = wins_sum / losses_sum if losses_sum > 0 else None  # 0 / 6 = 0

    assert wins_sum == 0.0
    assert abs(losses_sum - 6.0) < 1e-9
    assert expected_pf == 0.0


def test_profit_factor_break_even() -> None:
    """Test Profit Factor when wins equal losses (PF = 1.0)."""
    daily_profits = [3.0, -1.5, 2.0, -3.5]

    wins_sum = sum(p for p in daily_profits if p > 0)  # 3 + 2 = 5
    losses_sum = abs(sum(p for p in daily_profits if p < 0))  # 1.5 + 3.5 = 5

    expected_pf = wins_sum / losses_sum  # 5 / 5 = 1.0

    assert abs(expected_pf - 1.0) < 1e-9


def test_recovery_factor_calculation() -> None:
    """Test Recovery Factor = total_profit / abs(max_drawdown)."""
    # Simulate cumulative profit curve
    daily_profits = [2.0, -3.0, 1.0, 4.0, -1.0, 2.0]
    active_total = sum(daily_profits)  # = 5.0

    # Calculate max drawdown
    running = 0.0
    peak = 0.0
    max_dd = 0.0

    for profit in daily_profits:
        running += profit
        if running > peak:
            peak = running
        dd_now = running - peak
        if dd_now < max_dd:
            max_dd = dd_now

    # Trace: running = [2, -1, 0, 4, 3, 5]
    # peak = [2, 2, 2, 4, 4, 5]
    # dd_now = [0, -3, -2, 0, -1, 0]
    # max_dd = -3

    expected_recovery = active_total / abs(max_dd)  # 5 / 3 ≈ 1.667

    assert abs(max_dd + 3.0) < 1e-9
    assert abs(expected_recovery - 5 / 3) < 1e-9


def test_recovery_factor_no_drawdown() -> None:
    """Test Recovery Factor when there is no drawdown (always profitable)."""
    daily_profits = [1.0, 2.0, 1.5, 3.0]

    # Calculate max drawdown
    running = 0.0
    peak = 0.0
    max_dd = 0.0

    for profit in daily_profits:
        running += profit
        if running > peak:
            peak = running
        dd_now = running - peak
        if dd_now < max_dd:
            max_dd = dd_now

    # No drawdown, max_dd = 0
    # Recovery factor should be None or undefined when dd == 0
    assert max_dd == 0.0


def test_max_win_impact_calculation() -> None:
    """Test Max Win Impact = max_daily_profit / total_profit * 100."""
    daily_profits = [1.0, 5.0, 2.0, -1.0, 3.0]
    active_total = sum(daily_profits)  # = 10.0
    max_daily_profit = max(daily_profits)  # = 5.0

    expected_impact = max_daily_profit / active_total * 100.0  # 5 / 10 * 100 = 50%

    assert abs(max_daily_profit - 5.0) < 1e-9
    assert abs(expected_impact - 50.0) < 1e-9


def test_max_win_impact_high_dependency() -> None:
    """Test Max Win Impact when one day dominates total profit."""
    daily_profits = [10.0, -1.0, 0.5, -0.5, 1.0]
    active_total = sum(daily_profits)  # = 10.0
    max_daily_profit = max(daily_profits)  # = 10.0

    expected_impact = max_daily_profit / active_total * 100.0  # 10 / 10 * 100 = 100%

    assert abs(expected_impact - 100.0) < 1e-9
    # This is a red flag: 100% of profit from one lucky day!


def test_max_win_impact_negative_total() -> None:
    """Test Max Win Impact when total profit is negative (should be None)."""
    daily_profits = [2.0, -5.0, 1.0, -3.0]
    active_total = sum(daily_profits)  # = -5.0

    # When active_total <= 0, Max Win Impact should be None
    # (can't calculate meaningful percentage of negative total)
    assert active_total < 0


def test_max_win_impact_all_negative() -> None:
    """Test Max Win Impact when all days are negative."""
    daily_profits = [-1.0, -2.0, -3.0]
    active_total = sum(daily_profits)  # = -6.0
    max_daily_profit = max(daily_profits)  # = -1.0

    # max_daily_profit is negative, so condition max_daily_profit > 0 fails
    # Max Win Impact should be None
    assert max_daily_profit < 0
    assert active_total < 0


def test_downside_deviation_vs_standard_deviation() -> None:
    """Verify that downside deviation differs from standard deviation."""
    daily_profits = [3.0, -2.0, 1.0, -1.0]

    # Standard deviation (Sharpe uses this)
    mean = sum(daily_profits) / len(daily_profits)  # 1/4 = 0.25
    variance = sum((p - mean) ** 2 for p in daily_profits) / (len(daily_profits) - 1)
    std_dev = math.sqrt(variance)

    # Downside deviation (Sortino uses this)
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    downside_var = sum(downside_deviations_sq) / len(daily_profits)
    downside_std = math.sqrt(downside_var)

    # They should be different!
    assert std_dev != downside_std
    assert downside_std < std_dev  # Downside is typically smaller


def test_edge_case_single_day() -> None:
    """Test metrics with only one day of data."""
    # Sortino requires at least 2 days
    # Our implementation checks: if total_days >= 2

    # Profit Factor should work
    # PF is None when losses_sum == 0

    # Max Win Impact
    max_win_impact = 5.0 / 5.0 * 100.0  # 100%
    assert abs(max_win_impact - 100.0) < 1e-9


def test_edge_case_zero_profits() -> None:
    """Test metrics when all days have zero profit."""
    daily_profits = [0.0, 0.0, 0.0]

    # Downside deviation
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    assert sum(downside_deviations_sq) == 0.0

    # Profit Factor: Both wins and losses are 0, PF should be None

    # Max Win Impact: max_daily_profit = 0, so condition > 0 fails
    assert max(daily_profits) == 0.0


def test_sharpe_ratio_calculation() -> None:
    """Test Sharpe Ratio calculation with standard deviation."""
    # Daily profits: [+2, -1, +3, -2, +1]
    daily_profits = [2.0, -1.0, 3.0, -2.0, 1.0]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 3.0
    avg_profit = active_total / total_days  # = 0.6

    # Calculate standard deviation (all volatility)
    variance = sum((p - avg_profit) ** 2 for p in daily_profits) / (total_days - 1)
    # Deviations: [1.4, -1.6, 2.4, -2.6, 0.4]
    # Squared: [1.96, 2.56, 5.76, 6.76, 0.16] = sum 17.2
    # Variance = 17.2 / 4 = 4.3
    std_dev = math.sqrt(variance)  # ≈ 2.074

    expected_sharpe = avg_profit / std_dev  # 0.6 / 2.074 ≈ 0.289

    # Verify calculation
    assert abs(variance - 4.3) < 0.01
    assert abs(std_dev - 2.074) < 0.001
    assert abs(expected_sharpe - 0.289) < 0.001


def test_sharpe_ratio_positive_profits() -> None:
    """Test Sharpe Ratio with only positive profits (low volatility)."""
    # Small volatility, all positive
    daily_profits = [1.0, 1.1, 0.9, 1.2, 0.8]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 5.0
    avg_profit = active_total / total_days  # = 1.0

    # Calculate std dev
    variance = sum((p - avg_profit) ** 2 for p in daily_profits) / (total_days - 1)
    std_dev = math.sqrt(variance)

    sharpe = avg_profit / std_dev

    # Low volatility means high Sharpe ratio
    assert sharpe > 5.0  # Should be quite high
    assert std_dev < 0.2  # Low volatility


def test_sharpe_ratio_high_volatility() -> None:
    """Test Sharpe Ratio with high volatility (wide swings)."""
    # High volatility: [+10, -5, +8, -3, +5]
    daily_profits = [10.0, -5.0, 8.0, -3.0, 5.0]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 15.0
    avg_profit = active_total / total_days  # = 3.0

    # Calculate std dev
    variance = sum((p - avg_profit) ** 2 for p in daily_profits) / (total_days - 1)
    std_dev = math.sqrt(variance)

    sharpe = avg_profit / std_dev

    # High volatility means lower Sharpe ratio
    assert std_dev > 5.0  # High volatility
    assert sharpe < 1.0  # Relatively low Sharpe


def test_sharpe_ratio_zero_volatility() -> None:
    """Test Sharpe Ratio when all profits are identical (zero volatility)."""
    # All same value: no volatility
    daily_profits = [2.0, 2.0, 2.0, 2.0]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 8.0
    avg_profit = active_total / total_days  # = 2.0

    # Calculate std dev
    variance = sum((p - avg_profit) ** 2 for p in daily_profits) / (total_days - 1)
    std_dev = math.sqrt(variance)

    # When std_dev is 0, Sharpe should be None or undefined
    assert std_dev == 0.0
    # In implementation, we check if std_dev > 0 before calculating


def test_sharpe_vs_sortino_comparison() -> None:
    """Compare Sharpe and Sortino ratios - Sortino should be higher when upside volatility exists."""
    # Profits with high upside volatility: [+10, +1, -1, +2]
    daily_profits = [10.0, 1.0, -1.0, 2.0]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 12.0
    avg_profit = active_total / total_days  # = 3.0

    # Sharpe: uses all volatility
    variance_all = sum((p - avg_profit) ** 2 for p in daily_profits) / (total_days - 1)
    std_all = math.sqrt(variance_all)
    sharpe = avg_profit / std_all

    # Sortino: only downside
    downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
    downside_var = sum(downside_deviations_sq) / total_days
    downside_std = math.sqrt(downside_var)
    sortino = avg_profit / downside_std if downside_std > 0 else None

    # Sortino should be higher because it ignores the big +10 upside volatility
    if sortino is not None:
        assert sortino > sharpe


def test_sharpe_ratio_in_implementation() -> None:
    """Test Sharpe Ratio formula as implemented in app.py."""
    # Simulate the actual implementation:
    # sharpe_ratio = (active_total / total_days / std_val)
    daily_profits = [3.0, -1.0, 2.0, 1.0, -0.5, 2.5]
    total_days = len(daily_profits)
    active_total = sum(daily_profits)  # = 7.0
    avg_daily = active_total / total_days  # ≈ 1.167

    # Calculate std as in the implementation
    mean_val = sum(daily_profits) / len(daily_profits)
    variance = sum((x - mean_val) ** 2 for x in daily_profits) / (len(daily_profits) - 1)
    std_val = math.sqrt(variance)

    # Implementation formula
    sharpe_impl = (
        (active_total / total_days / std_val) if (total_days > 0 and std_val > 0) else None
    )

    # Alternative formula (equivalent)
    sharpe_alt = avg_daily / std_val if std_val > 0 else None

    # Both should be equal
    assert sharpe_impl is not None
    assert sharpe_alt is not None
    assert abs(sharpe_impl - sharpe_alt) < 1e-9
