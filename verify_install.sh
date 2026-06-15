#!/usr/bin/env bash
#
# Clean-install verification for the Monte Carlo risk lab.
# Run from the repository root inside a fresh virtual environment.
#
#   python -m venv .venv && source .venv/bin/activate   # (Linux/macOS)
#   ./verify_install.sh
#
set -euo pipefail

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing dependencies from requirements.txt"
pip install -r requirements.txt

echo "==> Running test suite"
python -m pytest -q

echo "==> Import check"
python -c "import app; import mc_core; import mc_report; print('imports ok')"

echo "==> CLI smoke test"
python monte_carlo_gbm.py AAPL --paths 1000 --horizon 10 --no-chart

echo "==> Million-path runner smoke test (chunk-safe)"
python run_gbm_million.py --ticker AAPL --paths 10000 --horizon 252 --years 5 --seed 42 --no-chart

echo "==> Clean-install verification complete."
