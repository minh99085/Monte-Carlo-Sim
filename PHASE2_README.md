# Phase 2 — Production Tactical MC Tester (Complete)

Phase 2 turns the short-horizon sketch into a **production-grade tactical Monte
Carlo rule tester**, while wiring remaining core enhancements (variance
reduction in the simulate loop, calibration, statistical backtests).

---

## Status

| Area | Status |
| --- | --- |
| Flexible rule engine (TP, trail, re-entry, callable entry) | **Done** |
| Historical rolling-window rule mode | **Done** |
| MC path generation via `mc_core.simulate` (full short paths) | **Done** |
| Sobol QMC + control variate in simulate loop | **Done** (Sobol needs SciPy; graceful fallback) |
| `mc_calibration.py` (GARCH MLE / Heston moments) | **Done** |
| Kupiec POF + rolling VaR coverage | **Done** |
| CLI `--tactical` | **Done** |
| GUI **Tactical** tab | **Done** |
| Optional Numba | Detected / reported (`numba_available` in stats); hot path uses NumPy vectorization |
| Plotly / PDF exports | **Phase 3** |
| Parallel multi-process path farm | **Phase 3** |

---

## How trading-rule testing works

```
TacticalConfig + TradingRule
        │
        ▼
mc_core.simulate  ── short paths (5–10d), optional VR
        │
        ▼
apply_rule_to_paths  ── entry / stop / TP / trail / max hold / re-entry
        │
        ▼
TacticalResult stats + optional historical + VaR backtest
```

### Executable rule features

| Feature | Field | Behavior |
| --- | --- | --- |
| Side | `side` or text `"short"` | Long or short |
| Entry day | `entry_day` | First eligible bar (default 0) |
| Callable entry | `entry_fn(day, prices_so_far)` | Custom signal |
| Stop loss | `stop_loss_pct` | Hard stop from entry |
| Take profit | `take_profit_pct` | Profit target |
| Trailing stop | `trailing_stop_pct` | From running favorable extreme |
| Max hold | `max_holding_days` | Time stop |
| Re-entry | `allow_reentry`, `max_trades` | Multiple round-trips per path |

Costs are applied on each entry and exit (same spirit as `mc_core.apply_costs`).

### Historical mode

```python
result = run_tactical_simulation(cfg, historical_prices=price_array, run_var_backtest=True)
print(result.historical.stats)
print(result.backtest["kupiec"])
```

Sliding windows of length `horizon+1` apply the **same** rule engine used on MC paths.

---

## Outputs

`TacticalResult.summary_text()` and `.to_stats_dict()` include:

- Chance of profit / loss / flat  
- Avg / median / best / worst P&L (+ percentiles)  
- Stop / TP / trailing / max-hold exit rates  
- Avg trades per path and holding days  
- Optional historical comparison and Kupiec VaR coverage  

---

## Examples

### Python

```python
from tactical_config import preset_5_day, TradingRule
from tactical_simulator import run_tactical_simulation

rule = TradingRule(
    name="Long 5d, 2% stop, 3% TP",
    entry_condition="Enter long at start",
    exit_condition="TP, stop, or max hold",
    stop_loss_pct=0.02,
    take_profit_pct=0.03,
    max_holding_days=5,
    side="long",
)
cfg = preset_5_day(
    "AAPL", paths=20_000, seed=42,
    starting_price=100.0, annual_volatility=0.25, annual_drift=0.0,
).with_rule(rule)

result = run_tactical_simulation(cfg)
print(result.summary_text())
```

### CLI

```powershell
python monte_carlo_gbm.py AAPL --tactical --paths 20000 --tactical-horizon 5 `
  --seed 42 --start-price 100 --sigma 0.25 --tactical-stop 0.02 --tactical-tp 0.03 --no-chart
```

### GUI

```powershell
streamlit run app.py
```

Open the **Tactical** tab, set horizon / stop / TP / trail / re-entry, click
**Run tactical simulation**. Results stay in `st.session_state` for downloads.

### Calibration (optional SciPy)

```python
import mc_calibration, numpy as np
rets = np.diff(np.log(prices))
g = mc_calibration.calibrate_garch(rets)
h = mc_calibration.calibrate_heston(rets)
# Pass g.as_config_kwargs() / h.as_config_kwargs() into SimulationConfig
```

### Variance reduction in the core engine

```python
cfg = SimulationConfig(..., variance_reduction="control_variate")  # GBM mean
cfg = SimulationConfig(..., variance_reduction="sobol")            # needs SciPy
result = simulate(cfg)
print(result.stats.get("variance_reduction_effective"))
print(result.stats.get("control_variate_mean"))
```

---

## Key modules

| File | Role |
| --- | --- |
| `tactical_config.py` | Presets + structured `TradingRule` |
| `tactical_simulator.py` | Path gen, flexible rule engine, historical mode, stats |
| `mc_calibration.py` | GARCH / Heston calibration helpers |
| `mc_core.py` | Engine + Sobol/CV wiring + Kupiec + rolling VaR |
| `monte_carlo_gbm.py` | `--tactical` CLI |
| `app.py` | **Tactical** GUI tab |
| `test_tactical.py` | Phase 2 unit tests |

---

## Phase 3 gaps (not in this drop)

- Plotly interactive charts and PDF report export  
- Multi-process / joblib path farms for multi-million tactical runs  
- Richer signal DSL (indicators, multi-asset rules)  
- Live data streaming / paper-trade bridge (out of scope for research tool)  
- Full options-surface Heston calibration  

---

## Disclaimer

Statistical research tool only. Not investment advice. No live order routing.
