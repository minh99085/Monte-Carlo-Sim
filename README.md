# Monte-Carlo-Sim

A CPU-first, memory-safe **Monte Carlo Geometric Brownian Motion (GBM)** simulator
for single-asset price/risk analysis. It ships with both a command line tool and a
Streamlit GUI and is designed to run comfortably on a Windows 11 laptop
(e.g. Intel i7-8750H, 16 GB RAM, **no GPU required**).

> Risk/research tool only. It does **not** connect to any live trading or
> order-routing system.

## Highlights

- **Memory safe by design** — a 1,000,000+ path simulation **never** allocates a
  full `paths × steps` matrix. Paths are produced in chunks and evolved one step
  at a time; only the per-path final values, aggregate statistics, and a small
  number of sample trajectories are retained.
- **Path modes** — Preview (10,000), Standard (100,000), Serious (1,000,000), and
  Tail-risk advanced (2,000,000–5,000,000, with a warning).
- **Risk metrics** — expected/median ending value, probability of profit/loss,
  Value at Risk (95 / 99 / 99.9), Expected Shortfall (95 / 99 / 99.9), and a full
  percentile table.
- **Transaction cost / slippage** modeling.
- **Exports** — CSV summary and JSON report.
- **Offline friendly** — market data is fetched via `yfinance` when available and
  falls back to safe default parameters when offline.

## Project layout

| File | Purpose |
| --- | --- |
| `mc_core.py` | Shared simulation engine (chunked GBM, statistics, VaR/ES, exports). |
| `monte_carlo_gbm.py` | Command line interface. |
| `app.py` | Streamlit GUI entry point. |
| `test_monte_carlo_gbm.py` | Test suite. |
| `requirements.txt` | Dependencies. |

## Windows (PowerShell) quick start

Run these commands in **Windows PowerShell**:

```powershell
# 1. Clone the repository
git clone https://github.com/minh99085/Monte-Carlo-Sim.git
cd Monte-Carlo-Sim

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
# (If activation is blocked by execution policy, run once:)
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 4. Install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the tests
python -m pytest -q

# 6. Run the CLI
python monte_carlo_gbm.py AAPL --paths 1000 --horizon 10 --no-chart

# 7. Launch the GUI
streamlit run app.py
```

To deactivate the virtual environment later, run `deactivate`.

## Command line usage

```powershell
python monte_carlo_gbm.py TICKER [options]
```

Common options:

| Option | Description | Default |
| --- | --- | --- |
| `--paths` | Number of simulated paths | `100000` |
| `--horizon` | Forward steps (trading days) | `252` |
| `--years` | Years of history for estimating `mu`/`sigma` | `3` |
| `--start-price` / `--s0` | Override the starting price | market close |
| `--mu` / `--sigma` | Override annualized drift / volatility | estimated |
| `--chunk-size` | Paths simulated per chunk | `50000` |
| `--seed` | Random seed (reproducibility) | random |
| `--cost` | Proportional transaction cost / slippage | `0.0` |
| `--export-csv PATH` | Write a CSV summary | — |
| `--export-json PATH` | Write a JSON report | — |
| `--no-chart` | Do not render charts | off |
| `--chart-path PATH` | Save the chart to a PNG | — |

Examples:

```powershell
# Standard run with reproducible seed
python monte_carlo_gbm.py MSFT --paths 100000 --horizon 252 --seed 7

# Serious 1,000,000-path run (chunk-safe) with exports
python monte_carlo_gbm.py TSLA --paths 1000000 --chunk-size 50000 --seed 7 `
  --export-csv tsla_summary.csv --export-json tsla_report.json --no-chart
```

## GUI usage

```powershell
streamlit run app.py
```

The sidebar lets you set the ticker, starting-price override, path mode, horizon,
chunk size, seed, and transaction cost. Results include headline metrics, a
VaR/Expected Shortfall table, a percentile table, sample-path and ending-value
charts, a convergence chart, runtime/memory status, and CSV/JSON download buttons.

## Simulation models

Beyond the GBM baseline, the engine (`mc_core.py`) supports several more
realistic models, all sharing the same chunk-safe execution:

| Model | Description | Key inputs |
| --- | --- | --- |
| **GBM Normal** | Classic geometric Brownian motion (normal shocks). | — |
| **Student-t GBM** | GBM with fat-tailed Student-t shocks (variance-normalized). | degrees of freedom |
| **Historical Bootstrap** | Samples empirical daily returns with replacement. | history |
| **Block Bootstrap** | Samples consecutive blocks to keep volatility clustering. | block length (default 20) |
| **Merton Jump-Diffusion** | GBM plus Poisson jumps (crypto preset is jump-heavy). | intensity, jump mean/vol |
| **Regime Switching** | Normal / high-vol / crash regimes via a transition matrix. | stock or crypto preset |

**Conservative drift mode**: choose Historical, Half historical, Zero, or Manual
drift. **Stress overlay** (optional, works on top of any model): a one-day crash
%, a volatility multiplier, and a drift haircut.

Extra output metrics include P(ending > +20%), P(ending < -10%),
P(ending < -20%), probability of a 50% drawdown, and the worst-1% average ending
value. CSV/JSON exports carry the model type, all model parameters, drift mode,
stress settings, probability buckets, and memory/chunk-safety info.

Examples:

```powershell
# Student-t fat tails, conservative (zero) drift
python monte_carlo_gbm.py AAPL --model "Student-t GBM" --t-df 4 --drift-mode "Zero drift" --no-chart

# Crypto-style jump diffusion with a stress overlay
python monte_carlo_gbm.py BTC-USD --model "Merton Jump-Diffusion" --crypto-jumps `
  --stress --stress-crash 0.2 --stress-vol-mult 1.5 --no-chart
```

## Model comparison

The GUI has a **Model comparison** tab: pick any subset of the six models (all
selected by default) and run them on identical ticker/paths/horizon/chunk/seed/
drift settings. The comparison table shows, per model, expected and median
ending value, P(profit)/P(loss), P(gain > 20%), P(loss > 10%), P(loss > 20%),
P(50% drawdown), the 5th/95th percentiles, 99% VaR and 99% Expected Shortfall,
runtime, and chunk-safe status. It highlights the **most conservative model**
(highest P(loss > 20%), then highest 99% Expected Shortfall) and exports the
comparison as **CSV** and **JSON** (the JSON includes per-model assumptions and
memory/chunk-safety metadata). Every comparison run uses the same chunk-safe
engine, so no full path × step matrix is ever allocated.

## Memory safety notes

- Default **Serious-mode chunk size is 25,000–50,000 paths**, which bounds peak
  working memory regardless of total path count.
- The engine stores final values (one 1-D array), aggregate statistics, and only
  a limited number of sample paths — it never materializes the full path matrix.
- The reported "memory safety" status compares actual peak working memory against
  what a naive full-matrix run would require.

## Running tests

```powershell
python -m pytest -q
```
