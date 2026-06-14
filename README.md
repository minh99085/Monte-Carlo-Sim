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
