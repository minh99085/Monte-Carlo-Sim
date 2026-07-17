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
| `mc_core.py` | Shared simulation engine (chunked models, statistics, VaR/ES, VR, backtests). |
| `monte_carlo_gbm.py` | Command line interface (includes `--tactical` mode). |
| `app.py` | Streamlit GUI entry point (includes **Tactical** tab). |
| `tactical_config.py` | Short-horizon presets + structured trading rules (Phase 1/2). |
| `tactical_simulator.py` | Tactical rule tester on MC paths + historical windows (Phase 2). |
| `mc_calibration.py` | Optional GARCH/Heston calibration helpers. |
| `mc_report.py` | Investment report engine. |
| `test_monte_carlo_gbm.py` / `test_tactical.py` | Test suites. |
| `PHASE1_README.md` / `PHASE2_README.md` | Phase docs. |
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

# 6b. Tactical short-horizon rule test (Phase 2)
python monte_carlo_gbm.py AAPL --tactical --paths 20000 --tactical-horizon 5 `
  --seed 42 --start-price 100 --sigma 0.25 --tactical-stop 0.02 --tactical-tp 0.03 --no-chart

# 7. Launch the GUI (use the Tactical tab for rule testing)
streamlit run app.py
```

To deactivate the virtual environment later, run `deactivate`.

## Clean-install verification

To confirm a fresh clone installs and runs end-to-end, use a brand-new virtual
environment and run the exact commands below.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python -c "import app; import mc_core; import mc_report; print('imports ok')"
python monte_carlo_gbm.py AAPL --paths 1000 --horizon 10 --no-chart
python run_gbm_million.py --ticker AAPL --paths 10000 --horizon 252 --years 5 --seed 42 --no-chart
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
./verify_install.sh
```

`verify_install.sh` runs the same upgrade-pip → install → tests → imports → CLI →
million-path checks in one step.

## Tactical short-horizon mode (Phase 2)

Test a trading rule on thousands of **5–10 trading day** Monte Carlo paths
(stop-loss, take-profit, trailing stop, max hold, optional re-entry), with
optional historical rolling-window validation and Kupiec VaR coverage.

```powershell
# CLI
python monte_carlo_gbm.py MSFT --tactical --paths 20000 --tactical-horizon 10 `
  --tactical-side short --tactical-stop 0.02 --tactical-trail 0.015 --no-chart

# Python API
python -c "from tactical_config import preset_5_day, TradingRule; from tactical_simulator import run_tactical_simulation; r=run_tactical_simulation(preset_5_day('AAPL', paths=5000, starting_price=100, annual_volatility=0.25)); print(r.summary_text())"
```

See `PHASE2_README.md` for the full rule engine, historical mode, calibration,
and variance-reduction notes. GUI: open the **Tactical** tab after
`streamlit run app.py`.

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

Advanced stochastic models are also available:

| Model | Description | Key inputs |
| --- | --- | --- |
| **Heston Stochastic Volatility** | Mean-reverting stochastic variance, full-truncation Euler (variance never goes negative). | kappa, theta, xi, rho, v0 |
| **GARCH(1,1)** | Volatility clustering via `sigma_t^2 = omega + alpha r^2 + beta sigma^2`. | omega, alpha, beta |
| **Kou Jump-Diffusion** | Asymmetric double-exponential jumps (better crash/upside asymmetry). | intensity, p_up, eta_up, eta_down |

**Conservative drift mode**: choose Historical, Half historical, Zero, or Manual
drift. **Stress overlay** (optional, works on top of any model): a one-day crash
%, a volatility multiplier, and a drift haircut.

### Advanced math (risk lab)

- **EVT tail risk**: a Generalized Pareto (peaks-over-threshold) fit to simulated
  losses gives EVT VaR/ES at 95/99/99.5/99.9 with the threshold, exceedance count,
  and a small-sample warning.
- **Variance reduction**: antithetic variates and a GBM control variate (CPU-only);
  optional Sobol QMC when scipy is present, with graceful fallback. Convergence
  diagnostics compare plain vs variance-reduced Monte Carlo. The method is recorded
  in exports as `variance_reduction_method`.
- **Advanced risk metrics**: max drawdown, drawdown duration, Calmar, annualized
  Sharpe/Sortino, probability of ruin (user threshold), and a *theoretical-only*
  Kelly fraction (with a strong warning).
- **Portfolio mode**: multi-asset correlated GBM with covariance shrinkage
  (Ledoit-Wolf when scikit-learn is available, else a diagonal-shrinkage fallback)
  and a Cholesky factor with PD repair. Outputs portfolio VaR/ES, per-asset summary,
  and a correlation-matrix export. Chunk-safe (per-asset block, never paths × steps).
- **Model validation**: a rolling-window GBM backtest reports the coverage of
  realized forward returns inside the model's 5–95 band, plus calibration warnings
  (very high drift, short history, low EVT exceedance count).
- Exports include `math_model_version`, all model parameters, drift mode, variance
  reduction method, a `warnings` list, and EVT/portfolio metadata when used. The
  comparison dashboard adds EVT 99% tail loss/ES, max-drawdown probability,
  probability of ruin, Sharpe/Sortino, and a model-disagreement rank.

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

## Investment Report (plain-English risk lab)

The GUI has an **Investment Report** tab that turns the simulation into an
institutional-style, plain-English risk report for a non-coder investor. You set
a ticker, horizon, risk tolerance (Conservative/Moderate/Aggressive), maximum
acceptable loss, ruin threshold, investment amount, and an optional benchmark
(default SPY), then click **Run Report**. It:

- runs the **full institutional model stack** (all nine models) on the chunk-safe
  engine, plus a **stress-test suite** (zero drift, half drift, doubled
  volatility, one-day −10%/−20% crashes, a bear regime, and a combined stress);
- computes a **model-risk confidence** score (High / Medium / Low) from history
  length, drift/volatility extremeness, model disagreement, tail severity, and
  backtest coverage — with a "do not rely on this result alone" warning when Low;
- gives a descriptive **investment label** (`Favorable but risky`,
  `Neutral / uncertain`, `Unfavorable risk-reward`, or `Too uncertain to judge`)
  — never a guaranteed buy/sell;
- shows **plain-English risk cards** (Profit Chance, Big Loss Chance, Severe
  Drawdown Chance, Worst Model, Model Confidence, Investment Label) and a written
  report (Bottom Line, What the Simulation Says, Best/Base/Bad/Severe-Stress
  cases, Biggest Risks, What Could Make This Wrong, Position Sizing Warning, and
  Suggested Next Questions);
- adds a **benchmark comparison** (beta, correlation, excess return, drawdown,
  downside risk) and best-effort **fundamentals** (gracefully shows
  "Fundamental data unavailable from source" when missing);
- exports a **Markdown** report, a **JSON** institutional report, and a **CSV**
  comparison table to `outputs/{ticker}_investment_report.md` / `.json` /
  `{ticker}_institutional_comparison.csv`.

This is a statistical risk simulation, not investment advice, a forecast, or a
guarantee.

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
