# Phase 2 — Testing Trading Rules on Simulated Paths

This document explains **Phase 2** of the short-horizon tactical trading
simulator built on top of
[Monte-Carlo-Sim](https://github.com/minh99085/Monte-Carlo-Sim).

Phase 1 gave you **settings** and a **rule checklist**.  
Phase 2 makes those rules **run against thousands of possible price stories**.

---

## What Phase 2 adds

| New file | Purpose |
| --- | --- |
| `tactical_simulator.py` | Generate short paths + apply rules + summarize results |
| `PHASE2_README.md` | This guide |

Still unchanged: `mc_core.py`, `app.py`, the CLI, and other original files.  
Phase 2 **reuses** the engine; it does not rewrite it.

---

## How trading-rule testing works

### Big picture

```
TacticalConfig + TradingRule   (from Phase 1)
        │
        ▼
mc_core.simulate(...)          ← invent many 5–10 day price paths
        │
        ▼
For each path, day by day:     ← apply entry / stop / max hold
        │
        ▼
Per-path P&L, stop hits, trades
        │
        ▼
Distribution summary           ← chance of profit, avg P&L, worst loss, …
```

### Step 1 — Generate short price paths

The simulator builds a `SimulationConfig` from your `TacticalConfig` and calls
the existing **`mc_core.simulate`** function.

For short horizons it asks the engine to **keep every full path**
(`sample_paths = paths`). That is cheap when the window is only 5–10 days:

- 100,000 paths × 6 prices ≈ a few megabytes  
- Long enough to walk day by day, small enough for a laptop  

Each path looks like:

| Index | Meaning |
| --- | --- |
| `prices[0]` | Starting price (“now”) |
| `prices[1]` | Close after 1 trading day |
| `prices[2]` | Close after 2 trading days |
| … | … |
| `prices[H]` | Close after H trading days (horizon) |

### Step 2 — Apply the rule on each path

Phase 1 stores entry/exit as plain English, plus two hard numbers:

- **`stop_loss_pct`** (e.g. `0.02` = 2%)  
- **`max_holding_days`** (e.g. `5`)

Phase 2 turns that into executable logic:

1. **Side (long vs short)**  
   - If the word `"short"` appears in `entry_condition` → **short**  
   - Otherwise → **long**

2. **Entry**  
   - Enter once at the **start of the window** (`prices[0]`).  
   - Matches the default Phase 1 language:  
     *“Enter long at the start of day 1”*.

3. **Each following day** `d = 1 … max_holding_days` (capped by the horizon):
   - Read that day’s close.
   - **Stop loss**
     - Long: stop if close ≤ entry × (1 − stop%)  
     - Short: stop if close ≥ entry × (1 + stop%)  
   - If the stop hits → exit that day, mark `stop_hit = True`.
   - If day `d` is the last allowed hold day → **time stop** exit  
     (`exit_reason = max_holding`).

4. **P&L** (per share), after proportional transaction costs on entry and exit  
   (same spirit as `mc_core.apply_costs`):
   - Long:  `exit×(1−cost) − entry×(1+cost)`  
   - Short: `entry×(1−cost) − exit×(1+cost)`

5. **Trades**  
   - Default rules complete **one** round-trip per path → `n_trades = 1`.

> Note: the free-text `exit_condition` is kept as documentation in Phase 2.  
> The **numbers that actually fire** are stop-loss and max holding period.

### Step 3 — Summarize all paths

Across every path the simulator computes a clear distribution:

| Metric | Meaning |
| --- | --- |
| Chance of profit / loss / flat | Fraction of paths with P&L > 0, < 0, ≈ 0 |
| Average / median P&L | Typical outcome (cash and %) |
| Best / worst P&L | Right and left extremes |
| 5th / 95th percentile P&L | Tail sketch without full charts |
| Stop-loss hit rate | How often the stop ended the trade |
| Max-hold exit rate | How often the time stop ended the trade |
| Avg trades per path | Typically ~1.0 with the default single-entry rule |
| Avg holding days | How long positions stayed open on average |

---

## What outputs you get

The main return type is **`TacticalResult`**.

### Human-readable summary

```python
print(result.summary_text())
```

Prints ticker, horizon, rule, runtime, and the distribution table above.

### Per-path arrays (for your own analysis)

| Attribute | Content |
| --- | --- |
| `result.pnl` | Cash P&L per path |
| `result.pnl_pct` | P&L as a fraction of entry |
| `result.n_trades` | Trades completed on that path |
| `result.stop_hit` | `True` if the stop fired |
| `result.exit_reason` | `"stop_loss"` or `"max_holding"` |
| `result.holding_days` | Days the position was open |
| `result.price_paths` | Full price matrix `(paths × horizon+1)` |

### Compact stats dict

```python
stats = result.to_stats_dict()
# e.g. stats["prob_profit"], stats["avg_pnl"], stats["worst_pnl"], …
```

---

## Key functions

| Function | Role |
| --- | --- |
| `run_tactical_simulation(cfg)` | **Main entry point** — generate paths, apply rule, return `TacticalResult` |
| `generate_price_paths(cfg)` | Call `mc_core.simulate` and return the full price matrix |
| `build_simulation_config(cfg, …)` | Map Phase 1 settings → `SimulationConfig` (keeps all paths) |
| `resolve_market_parameters(cfg)` | Fill s0 / mu / sigma (manual, yfinance, or fallback) |
| `infer_side(rule)` | Long vs short from entry text |
| `apply_rule_to_one_path(prices, rule)` | Day-by-day walk for a single story (easy to read) |
| `apply_rule_to_paths(matrix, rule)` | Same logic for all paths (fast) |
| `compute_tactical_stats(outcomes)` | Build the distribution summary dict |

---

## Example: run a basic tactical simulation

### Option A — from Python

```python
from tactical_config import preset_5_day, TradingRule
from tactical_simulator import run_tactical_simulation

# 1) Start from the 5-trading-day preset (includes a default long rule + 2% stop)
cfg = preset_5_day(
    "AAPL",
    paths=20_000,              # enough for a smooth preview
    seed=42,                   # reproducible
    starting_price=100.0,      # optional; otherwise market/fallback
    annual_volatility=0.25,    # optional; otherwise estimated
    annual_drift=0.0,          # short-horizon conservative default
)

# 2) Run the tactical simulation
result = run_tactical_simulation(cfg)

# 3) Read the plain-English summary
print(result.summary_text())

# 4) Dig into numbers if you want
print("Profit chance:", result.stats["prob_profit"])
print("Average P&L:  ", result.stats["avg_pnl"])
print("Worst loss:   ", result.stats["worst_pnl"])
print("Stop hit rate:", result.stats["stop_hit_rate"])
```

### Option B — custom rule (including short)

```python
from tactical_config import preset_10_day, TradingRule
from tactical_simulator import run_tactical_simulation

rule = TradingRule(
    name="Short, 5-day hold, 2% stop",
    entry_condition="Enter short at the start of day 1",
    exit_condition="Cover at max hold or stop",
    stop_loss_pct=0.02,
    max_holding_days=5,
)

cfg = preset_10_day(
    "MSFT",
    paths=10_000,
    seed=7,
    starting_price=100.0,
    annual_volatility=0.30,
).with_rule(rule)

result = run_tactical_simulation(cfg)
print(result.summary_text())
```

### Option C — run the built-in demo from the terminal

```powershell
cd Monte-Carlo-Sim
python tactical_simulator.py
```

This runs a 5-day long demo and a small 10-day short-rule demo with fixed
prices/vol so it works offline.

---

## Tips for short horizons (5–10 days)

- Prefer **`preset_5_day`** / **`preset_10_day`** so horizon and max hold stay aligned.
- Use **`annual_drift=0.0`** if you do not want the model to assume a bullish trend over a few days.
- Start with **10k–50k paths** while exploring; use **100k** for smoother stats.
- Keep **`max_holding_days ≤ horizon_days`** (validation enforces this).
- Transaction cost matters more for short trades; the presets use **0.1%**.

---

## What Phase 2 does *not* do yet

- It does not parse complex free-text strategies (only side + stop + max hold).
- It does not re-enter after an exit (one trade per path).
- It does not add a GUI tab or CLI flag (you call it from Python).
- It does not place live orders.

Those are natural later phases (richer rule language, multi-trade paths,
reports, UI).

---

## Files after Phase 1 + Phase 2

| File | Role |
| --- | --- |
| `tactical_config.py` | Phase 1 — config presets + `TradingRule` |
| `tactical_simulator.py` | Phase 2 — path generation + rule testing + stats |
| `PHASE1_README.md` | Phase 1 docs |
| `PHASE2_README.md` | Phase 2 docs (this file) |
| `mc_core.py` | Original engine (reused, not modified) |

---

## Disclaimer

This is a **statistical research tool**. Simulated P&L is not a forecast, not
investment advice, and not a live trading system.
