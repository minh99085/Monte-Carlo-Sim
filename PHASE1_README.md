# Phase 1 — Short-Horizon Tactical Foundations

This document explains **Phase 1** of extending
[Monte-Carlo-Sim](https://github.com/minh99085/Monte-Carlo-Sim) into a
practical **short-horizon tactical trading simulator** (roughly **5–10
trading days**).

Phase 1 only **adds new files**. The original simulation engine and UI are
unchanged.

---

## What the original tool does today

The existing project creates many possible **future price paths** for a stock
(using models such as Geometric Brownian Motion) and turns those paths into
**risk numbers**: expected ending value, chance of profit/loss, Value at Risk,
and so on.

By default, the core engine is oriented toward a **long horizon**
(about **252 trading days** ≈ one year). That is useful for strategic risk
questions, but it is heavier than you need for **week-to-two-week** trade ideas.

---

## What Phase 1 adds

Phase 1 introduces two new building blocks:

| New file | Purpose |
| --- | --- |
| `tactical_config.py` | Easy short-horizon settings + a basic trading-rule structure |
| `PHASE1_README.md` | This guide (plain-English documentation) |

In plain language, Phase 1 gives you:

1. **Easy presets** for “simulate the next **5 trading days**” and “the next
   **10 trading days**”.
2. A **clean config object** (`TacticalConfig`) with defaults that make sense
   for short-term tactics (short horizon, optional zero drift, a small
   transaction cost, reproducible seed).
3. A **basic trading rule structure** (`TradingRule`) so you can write down:
   - when you would **enter**
   - when you would **exit**
   - where the **stop loss** sits
   - the **maximum holding period** in days
4. A small **bridge** (`to_simulation_kwargs()`) that maps tactical settings
   onto the field names already used by `SimulationConfig` in `mc_core.py`,
   ready for a later phase to plug in.

**What Phase 1 does *not* do yet**

- It does **not** change `mc_core.py`, `app.py`, or the CLI.
- It does **not** automatically execute entry/exit/stop logic on simulated paths.
- It does **not** place live trades or connect to a broker.

Those steps belong to later phases.

---

## Project layout (after Phase 1)

| File | Role |
| --- | --- |
| `mc_core.py` | Original shared simulation engine (unchanged) |
| `monte_carlo_gbm.py` | Original command-line tool (unchanged) |
| `app.py` | Original Streamlit GUI (unchanged) |
| **`tactical_config.py`** | **New** – short-horizon config + trading rules |
| **`PHASE1_README.md`** | **New** – this documentation |

---

## How to use the new tactical configuration

### 1. Import the helpers

```python
from tactical_config import (
    TacticalConfig,
    TradingRule,
    preset_5_day,
    preset_10_day,
    get_preset,
)
```

### 2. Use a preset (recommended starting point)

**5 trading days** (about one trading week):

```python
cfg = preset_5_day("AAPL")
print(cfg.summary())
```

**10 trading days** (about two trading weeks):

```python
cfg = preset_10_day("MSFT")
print(cfg.summary())
```

**Same thing by name:**

```python
cfg = get_preset("5d", ticker="AAPL")
cfg = get_preset("10", ticker="MSFT")
```

### 3. Override any setting you care about

```python
cfg = preset_5_day(
    "TSLA",
    paths=50_000,           # fewer paths = faster preview
    seed=7,                 # different reproducible seed
    transaction_cost=0.002, # 0.2% friction
    annual_volatility=0.40, # assume 40% annual vol
)
```

### 4. Build a config by hand (if you prefer)

```python
cfg = TacticalConfig(
    ticker="AAPL",
    horizon_days=5,
    paths=100_000,
    seed=42,
    annual_drift=0.0,          # do not assume upward drift over a few days
    transaction_cost=0.001,    # 0.1%
).validate()
```

### 5. Map onto the existing engine (preview only in Phase 1)

```python
# This only builds a dictionary of keyword arguments.
# It does not run a simulation by itself.
kwargs = cfg.to_simulation_kwargs()
print(kwargs)

# Later phases can do something like:
#   from mc_core import SimulationConfig
#   sim_cfg = SimulationConfig(**kwargs)
```

### 6. Quick self-check from the terminal

```powershell
python tactical_config.py
```

This prints a short summary of the 5-day and 10-day presets and an example
custom rule.

---

## Default values (and why they fit short-term use)

| Setting | Phase 1 default | Why |
| --- | --- | --- |
| `horizon_days` | `5` (or `10` in the 10-day preset) | Focus on one or two trading weeks |
| `paths` | `100_000` | Stable risk stats without a full “serious” million-path run |
| `seed` | `42` | Reproducible by default; set to `None` for a fresh random run |
| `annual_drift` | `0.0` | Over a few days, assumed growth often matters less than volatility |
| `transaction_cost` | `0.001` (0.1%) | Short trades feel costs more; a small friction is realistic |
| `sample_paths` | `40` | Enough paths for charts without using much memory |
| Default stop | `2%` | Common short-term risk budget on the starter rule |

Percent fields use **fractions**: `0.02` means **2%**, not “2”.

---

## How the basic trading rule structure works

### The idea

A `TradingRule` is a **simple checklist** for one tactical idea. It stores:

| Field | Meaning | Example |
| --- | --- | --- |
| `name` | Label for reports | `"Momentum long, 5-day hold"` |
| `entry_condition` | When to open the trade | `"Enter long at the start of day 1"` |
| `exit_condition` | Normal (non-stop) exit | `"Exit if gain reaches +3%, else at day 5"` |
| `stop_loss_pct` | Loss cut as a fraction of entry | `0.02` → exit if price falls **2%** |
| `max_holding_days` | Hard time stop in trading days | `5` |
| `notes` | Free-text thesis / comments | optional |

In Phase 1 these are **descriptions you can define, validate, and print**.
They are ready to be attached to a `TacticalConfig`. They are **not yet**
executed step-by-step on every Monte Carlo path.

### Example: use the default rule from a preset

```python
cfg = preset_5_day("AAPL")
print(cfg.trading_rule.summary())
```

The 5-day preset attaches a starter long rule:

- Enter at the start of day 1  
- Exit at the close of day 5  
- 2% stop loss  
- Max hold = 5 trading days  

### Example: write your own rule

```python
from tactical_config import TradingRule, preset_5_day

rule = TradingRule(
    name="Gap-fade short",
    entry_condition="Enter short if open gaps up more than 1%",
    exit_condition="Cover if price reverts to prior close, or at day 3",
    stop_loss_pct=0.015,   # 1.5% stop
    max_holding_days=3,
    notes="Illustration only — not a recommendation.",
).validate()

cfg = preset_5_day("TSLA").with_rule(rule)
print(cfg.summary())
```

### Validation rules (safety checks)

Calling `.validate()` (or using a preset / `with_rule`) will raise an error if:

- the name or entry/exit text is empty  
- `stop_loss_pct` is outside `[0, 1)`  
- `max_holding_days` is less than 1  
- the rule’s max hold is **longer** than the config’s `horizon_days`  

That last check keeps the trade idea inside the simulated window.

---

## How Phase 1 fits the existing codebase

```
User / notebook / future CLI
        │
        ▼
tactical_config.py          ← Phase 1 (new)
  • TacticalConfig
  • TradingRule
  • preset_5_day / preset_10_day
  • to_simulation_kwargs()
        │
        │  (later phases will connect here)
        ▼
mc_core.py                  ← existing engine (unchanged in Phase 1)
  • SimulationConfig
  • path generation + risk metrics
        │
        ▼
monte_carlo_gbm.py / app.py ← existing CLI / GUI (unchanged in Phase 1)
```

Main original files (for context only):

- **`mc_core.py`** — shared Monte Carlo engine, `SimulationConfig`, risk stats  
- **`monte_carlo_gbm.py`** — command-line entry point  
- **`app.py`** — Streamlit GUI  
- **`mc_report.py`** — investment-style reporting helpers  

---

## Suggested next phases (not implemented yet)

These are **not part of Phase 1**; they are only a roadmap:

1. **Run short-horizon sims from `TacticalConfig`** without hand-building
   `SimulationConfig`.
2. **Apply `TradingRule` on each path** (entry day, stop hits, time stop,
   exit condition) and collect trade-level P&amp;L.
3. **Tactical risk report** focused on 5–10 day outcomes (win rate, average
   winner/loser, stop-out rate, max adverse excursion).
4. Optional GUI / CLI flags for “Tactical 5-day” and “Tactical 10-day” modes.

---

## Disclaimer

This project is a **statistical risk / research tool**.  
Phase 1 configuration and trading-rule text are **not** investment advice,
signals, or a guarantee of future results. Nothing here places live orders.
