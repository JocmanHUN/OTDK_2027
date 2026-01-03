# Risk Metrics Test Coverage

## Áttekintés

A kockázati mutatók (Sharpe Ratio, Sortino Ratio, Profit Factor, Recovery Factor, Max Win Impact) számítását átfogó tesztekkel fedjük le.

## Teszt fájlok

### 1. `test_risk_metrics.py` - Unit tesztek
**22 teszt** a matematikai formulák helyességére:

#### Sharpe Ratio tesztek:
- ✅ `test_sharpe_ratio_calculation` - Normál számítás vegyes profitokkal
- ✅ `test_sharpe_ratio_positive_profits` - Alacsony volatilitás (magas Sharpe)
- ✅ `test_sharpe_ratio_high_volatility` - Magas volatilitás (alacsony Sharpe)
- ✅ `test_sharpe_ratio_zero_volatility` - Nulla volatilitás (std = 0)
- ✅ `test_sharpe_ratio_in_implementation` - Implementációs formula validálása
- ✅ `test_sharpe_vs_sortino_comparison` - Sharpe vs Sortino összehasonlítás

#### Sortino Ratio tesztek:
- ✅ `test_sortino_ratio_calculation` - Normál számítás vegyes profitokkal
- ✅ `test_sortino_ratio_all_positive` - Csak pozitív profitoknál (downside_std = 0)
- ✅ `test_sortino_ratio_all_negative` - Csak negatív profitoknál

#### Profit Factor tesztek:
- ✅ `test_profit_factor_calculation` - Normál számítás (wins/losses)
- ✅ `test_profit_factor_no_losses` - Nincs veszteség (PF = None/inf)
- ✅ `test_profit_factor_no_wins` - Nincs nyereség (PF = 0)
- ✅ `test_profit_factor_break_even` - Egyensúly (PF = 1.0)

#### Recovery Factor tesztek:
- ✅ `test_recovery_factor_calculation` - Normál számítás (profit/DD)
- ✅ `test_recovery_factor_no_drawdown` - Nincs zuhanás (RF = None)

#### Max Win Impact tesztek:
- ✅ `test_max_win_impact_calculation` - Normál számítás
- ✅ `test_max_win_impact_high_dependency` - Magas függőség (100%)
- ✅ `test_max_win_impact_negative_total` - Negatív összprofitnál
- ✅ `test_max_win_impact_all_negative` - Csak veszteséges napoknál

#### Edge case-ek:
- ✅ `test_downside_deviation_vs_standard_deviation` - DD vs SD különbség
- ✅ `test_edge_case_single_day` - 1 napos adat
- ✅ `test_edge_case_zero_profits` - Nulla profiток

---

### 2. `test_risk_metrics_integration.py` - Integrációs tesztek
**7 teszt** a teljes pipeline működésére:

#### Adatbázis integráció:
- ✅ `test_compute_daily_stats_with_mixed_profits` - Vegyes napok, DD számítás
- ✅ `test_compute_daily_stats_sortino_calculation` - Sortino a teljes pipeline-ban
- ✅ `test_compute_daily_stats_profit_factor` - PF több napos adatokkal
- ✅ `test_compute_daily_stats_max_win_impact` - Max Win egy domináló napnál
- ✅ `test_compute_daily_stats_recovery_factor` - Recovery számítás zuhanással
- ✅ `test_compute_daily_stats_sharpe_ratio` - Sharpe ratio teljes pipeline tesztelése
- ✅ `test_compute_daily_stats_no_data` - Üres adatbázis kezelése

---

## Lefedettség

### Tesztelt komponensek:

| Komponens | Lefedettség | Tesztek száma |
|-----------|-------------|---------------|
| **Sharpe Ratio** | ✅ 100% | 6 unit + 1 integráció |
| **Sortino Ratio** | ✅ 100% | 4 unit + 1 integráció |
| **Profit Factor** | ✅ 100% | 4 unit + 1 integráció |
| **Recovery Factor** | ✅ 100% | 2 unit + 1 integráció |
| **Max Win Impact** | ✅ 100% | 4 unit + 1 integráció |
| **Daily Stats Pipeline** | ✅ Magas | 7 integrációs teszt |

### Összesített eredmény:

```
tests/gui/test_risk_metrics.py .......................  22 passed
tests/gui/test_risk_metrics_integration.py .......       7 passed
========================================================
Total: 29 passed
```

---

## Tesztelt forgatókönyvek

### Matematikai helyesség:
- [x] **Sharpe Ratio:** Átlagos napi profit / standard deviation (összes volatilitás)
- [x] **Sortino Ratio:** Downside deviation csak negatív eltéréseket számolja
- [x] Sharpe vs Sortino különbség (upside volatilitás figyelmen kívül hagyása)
- [x] Profit Factor = sum(wins) / sum(losses)
- [x] Recovery Factor = profit / abs(max_drawdown)
- [x] Max Win Impact százalékos számítás

### Edge case-ek:
- [x] Nulla volatilitás (std = 0)
- [x] Alacsony vs magas volatilitás (Sharpe ratio változása)
- [x] Nincs veszteség (losses_sum = 0)
- [x] Nincs nyereség (wins_sum = 0)
- [x] Nincs zuhanás (max_dd = 0)
- [x] Negatív összprofit
- [x] 1 napos adat (std nem számolható)
- [x] Üres adatbázis

### Integrációs működés:
- [x] Adatbázis lekérdezések
- [x] Napi aggregálás
- [x] Kumulált profit számítás
- [x] Max drawdown követés
- [x] Többnapos profit pattern-ek

---

## Hogyan futtasd?

### Minden teszt:
```bash
python -m pytest tests/gui/ -v
```

### Csak risk metrics:
```bash
python -m pytest tests/gui/test_risk_metrics.py tests/gui/test_risk_metrics_integration.py -v
```

### Lefedettségi jelentés:
```bash
python -m pytest tests/gui/ --cov=src.gui --cov-report=term-missing
```

### Egy konkrét teszt:
```bash
python -m pytest tests/gui/test_risk_metrics.py::test_sortino_ratio_calculation -v
```

---

## Validált formulák

### 1. Sharpe Ratio
```python
mean_val = sum(daily_profits) / total_days
variance = sum((p - mean_val) ** 2 for p in daily_profits) / (total_days - 1)
std_dev = sqrt(variance)
sharpe = (total_profit / total_days) / std_dev  # avg_daily_profit / std
```

### 2. Sortino Ratio
```python
downside_deviations_sq = [min(0, p) ** 2 for p in daily_profits]
downside_var = sum(downside_deviations_sq) / total_days
downside_std = sqrt(downside_var)
sortino = avg_daily_profit / downside_std
```

### 3. Profit Factor
```python
wins_sum = sum(p for p in daily_profits if p > 0)
losses_sum = abs(sum(p for p in daily_profits if p < 0))
profit_factor = wins_sum / losses_sum  # if losses_sum > 0
```

### 4. Recovery Factor
```python
recovery_factor = total_profit / abs(max_drawdown)  # if max_dd != 0
```

### 5. Max Win Impact
```python
max_daily_profit = max(daily_profits)
largest_win_impact = (max_daily_profit / total_profit * 100.0)  # if total_profit > 0
```

---

## Következő lépések (opcionális)

Ha tovább akarod növelni a lefedettséget:

1. **VaR 95% teszt** - Value at Risk számítás
2. **Win/Loss Streak teszt** - Leghosszabb sorozatok
3. **Ulcer Index teszt** - Stressz mérés
4. **Calmar Ratio teszt** - Éves hozam / DD

Ezekhez hasonló teszteket írhatsz, mint a meglévőkhöz! 🚀
