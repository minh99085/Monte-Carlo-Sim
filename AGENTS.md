# AGENTS.md

## Cursor Cloud specific instructions

This is a **CPU-only Python Monte Carlo (GBM) risk simulator** with a CLI, a
Streamlit GUI, and a TradingView webhook bridge. There is no database or backend
service to provision; everything runs in a single Python virtual environment.

### Environment

- Use the virtual environment at `.venv` (created by the startup update script).
  The system `python` binary does not exist — use `python3` or, preferably,
  `.venv/bin/python` and `.venv/bin/pytest` / `.venv/bin/streamlit`.
- Dependencies come from `requirements.txt` (numpy, pandas, matplotlib, streamlit,
  yfinance, pytest).

### Lint / test / build / run (standard commands live in `README.md`)

- Tests: `.venv/bin/python -m pytest -q` (this is the gate required before pushes,
  per `CLAUDE.md`). ~22s, expect passes with some skips.
- CLI: `.venv/bin/python monte_carlo_gbm.py AAPL --paths 1000 --horizon 10 --no-chart`.
  Full option matrix and more examples are in `README.md`.
- GUI (the primary app): `.venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true`.
  Health check: `curl http://localhost:8501/_stcore/health`.
- There is no separate lint tool configured in the repo; correctness is enforced
  via the pytest suite.

### Non-obvious notes

- `yfinance` fetches market data online but the code **falls back to safe default
  parameters when offline**, so CLI/GUI runs still work without network access.
  Use `--start-price` / `--sigma` / `--mu` (or the GUI overrides) for deterministic,
  network-free runs.
- Generated artifacts (`*.png`, `*.csv`, `*.json`, `outputs/`, `tv_data/`,
  `calibration/`) are gitignored — do not commit them.
- The engine is intentionally chunk-safe: even million-path runs never allocate a
  full `paths × steps` matrix, so large `--paths` values are safe on limited RAM.
