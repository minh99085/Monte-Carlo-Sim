#!/usr/bin/env python3
"""Standalone 1,000,000-path GBM Monte Carlo runner.

Pulls historical daily prices (via ``yfinance``), estimates annualized drift and
volatility, runs a **chunked** Geometric Brownian Motion simulation that never
allocates a full ``paths x steps`` matrix, and writes a CSV summary, a JSON
report and a high-resolution PNG chart to ``outputs/``.

Example (Windows PowerShell)::

    python run_gbm_million.py --ticker AAPL --paths 1000000 --horizon 252 --years 5 --seed 42

The heavy lifting (chunked simulation, parameter estimation, statistics) is
shared with the CLI and GUI through :mod:`mc_core`, so all three entry points
stay consistent and the memory-safety guarantees are identical.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from typing import Any, Dict, Optional

import numpy as np

import mc_core

DEFAULT_OUTDIR = "outputs"
DEFAULT_GAIN_THRESHOLD = 0.20   # "gains more than 20%"
DEFAULT_DROP_THRESHOLD = 0.10   # "drops more than 10%"
MAX_SAMPLE_PATHS_ON_CHART = 100


# ---------------------------------------------------------------------------
# Pure, testable helpers
# ---------------------------------------------------------------------------


def probability_gain(final_values: np.ndarray, s0: float, threshold: float) -> float:
    """P(ending price gains more than ``threshold`` vs ``s0``)."""
    fv = np.asarray(final_values, dtype=float)
    return float(np.mean(fv > s0 * (1.0 + threshold)))


def probability_drop(final_values: np.ndarray, s0: float, threshold: float) -> float:
    """P(ending price drops more than ``threshold`` vs ``s0``)."""
    fv = np.asarray(final_values, dtype=float)
    return float(np.mean(fv < s0 * (1.0 - threshold)))


def build_config(
    *,
    ticker: str,
    s0: float,
    mu: float,
    sigma: float,
    paths: int = mc_core.SERIOUS_PATHS,
    horizon: int = 252,
    seed: Optional[int] = None,
    chunk_size: int = mc_core.DEFAULT_SERIOUS_CHUNK_SIZE,
    sample_paths: int = MAX_SAMPLE_PATHS_ON_CHART,
) -> mc_core.SimulationConfig:
    """Assemble a validated, chunk-safe :class:`mc_core.SimulationConfig`.

    ``sample_paths`` is capped at :data:`MAX_SAMPLE_PATHS_ON_CHART` so the chart
    never plots more than 100 trajectories and the retained sample block stays
    tiny regardless of the total path count.
    """

    return mc_core.SimulationConfig(
        ticker=ticker,
        s0=float(s0),
        mu=float(mu),
        sigma=float(sigma),
        paths=int(paths),
        horizon=int(horizon),
        seed=seed,
        chunk_size=int(chunk_size),
        sample_paths=min(int(sample_paths), MAX_SAMPLE_PATHS_ON_CHART),
    ).validate()


def summarize(
    result: mc_core.SimulationResult,
    market: Optional[mc_core.MarketParameters] = None,
    *,
    gain_threshold: float = DEFAULT_GAIN_THRESHOLD,
    drop_threshold: float = DEFAULT_DROP_THRESHOLD,
) -> Dict[str, Any]:
    """Build the report dict with exactly the required output fields."""

    cfg = result.config
    fv = result.final_values
    s0 = cfg.s0

    summary: Dict[str, Any] = {
        "ticker": cfg.ticker,
        "starting_price": s0,
        "paths": cfg.paths,
        "horizon": cfg.horizon,
        "mu_annual": cfg.mu,
        "sigma_annual": cfg.sigma,
        "chunk_size": cfg.chunk_size,
        "seed": cfg.seed,
        "average_ending_price": float(np.mean(fv)),
        "median_ending_price": float(np.median(fv)),
        "percentile_1": float(np.percentile(fv, 1)),
        "percentile_5": float(np.percentile(fv, 5)),
        "percentile_95": float(np.percentile(fv, 95)),
        "percentile_99": float(np.percentile(fv, 99)),
        "gain_threshold": gain_threshold,
        "drop_threshold": drop_threshold,
        "prob_gain_more_than_20pct": probability_gain(fv, s0, gain_threshold),
        "prob_drop_more_than_10pct": probability_drop(fv, s0, drop_threshold),
        "runtime_seconds": result.runtime_seconds,
        "chunk_safe": result.memory.is_chunk_safe,
        "peak_matrix_elements": result.memory.peak_matrix_elements,
        "full_matrix_elements": result.memory.full_matrix_elements,
        "memory_status": result.memory.status(),
    }
    if market is not None:
        summary["data_source"] = market.source
        if market.note:
            summary["data_note"] = market.note
    return summary


def summary_to_csv(summary: Dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value"])
    for key, value in summary.items():
        writer.writerow([key, value])
    return buffer.getvalue()


def summary_to_json(summary: Dict[str, Any], *, indent: int = 2) -> str:
    return json.dumps(summary, indent=indent, default=mc_core._json_default)


def write_outputs(
    result: mc_core.SimulationResult,
    summary: Dict[str, Any],
    *,
    outdir: str = DEFAULT_OUTDIR,
    make_chart: bool = True,
    dpi: int = 200,
) -> Dict[str, str]:
    """Write CSV, JSON and (optionally) a high-resolution PNG into ``outdir``.

    Returns a mapping of output kind -> file path.
    """

    os.makedirs(outdir, exist_ok=True)
    ticker = str(summary.get("ticker", "ASSET")).replace(os.sep, "_")
    paths_written: Dict[str, str] = {}

    csv_path = os.path.join(outdir, f"{ticker}_gbm_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(summary_to_csv(summary))
    paths_written["csv"] = csv_path

    json_path = os.path.join(outdir, f"{ticker}_gbm_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(summary_to_json(summary))
    paths_written["json"] = json_path

    if make_chart:
        png_path = os.path.join(outdir, f"{ticker}_gbm_chart.png")
        _render_chart(result, png_path, dpi=dpi)
        paths_written["png"] = png_path

    return paths_written


def _render_chart(result: mc_core.SimulationResult, png_path: str, *, dpi: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = result.config
    traj = result.sample_trajectories[:MAX_SAMPLE_PATHS_ON_CHART]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(traj.shape[0]):
        ax1.plot(traj[i], linewidth=0.6, alpha=0.6)
    ax1.set_title(
        f"{cfg.ticker}: {traj.shape[0]} sample GBM paths "
        f"(of {cfg.paths:,}), {cfg.horizon} steps"
    )
    ax1.set_xlabel("trading day")
    ax1.set_ylabel("price")
    ax1.axhline(cfg.s0, color="black", linestyle="--", linewidth=1, label="S0")
    ax1.legend(loc="upper left")

    ax2.hist(result.final_values, bins=120, color="#3b7dd8", alpha=0.85)
    ax2.axvline(cfg.s0, color="black", linestyle="--", linewidth=1, label="S0")
    ax2.set_title(f"Ending-price distribution ({cfg.paths:,} paths)")
    ax2.set_xlabel("ending price")
    ax2.set_ylabel("frequency")
    ax2.legend(loc="upper right")

    fig.suptitle(
        f"GBM Monte Carlo  -  mu={cfg.mu:.2%}/yr, sigma={cfg.sigma:.2%}/yr",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_gbm_million.py",
        description="Standalone chunked 1,000,000-path GBM Monte Carlo runner.",
    )
    p.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL).")
    p.add_argument("--paths", type=int, default=mc_core.SERIOUS_PATHS,
                   help="Number of simulated paths (default: 1,000,000).")
    p.add_argument("--horizon", type=int, default=252,
                   help="Forward trading-day steps (default: 252).")
    p.add_argument("--years", type=float, default=5.0,
                   help="Years of history used to estimate mu/sigma (default: 5).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--chunk-size", type=int, default=mc_core.DEFAULT_SERIOUS_CHUNK_SIZE,
                   help="Paths simulated per chunk (default: 50,000).")
    p.add_argument("--start-price", "--s0", dest="start_price", type=float, default=None,
                   help="Override the starting price S0.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR,
                   help="Directory for CSV/JSON/PNG outputs (default: outputs).")
    p.add_argument("--no-chart", action="store_true",
                   help="Skip rendering the PNG chart.")
    return p


def _print_summary(summary: Dict[str, Any]) -> None:
    def money(x):
        return f"{x:,.2f}"

    print("=" * 64)
    print(f" GBM Monte Carlo: {summary['ticker']}  ({summary['paths']:,} paths)")
    print("=" * 64)
    print(f" Data source            : {summary.get('data_source', 'n/a')}")
    print(f" Starting price S0      : {money(summary['starting_price'])}")
    print(f" Drift mu (annual)      : {summary['mu_annual']:.4%}")
    print(f" Volatility sigma (ann) : {summary['sigma_annual']:.4%}")
    print(f" Horizon (steps)        : {summary['horizon']}")
    print(f" Chunk size             : {summary['chunk_size']:,}")
    print("-" * 64)
    print(f" Average ending price   : {money(summary['average_ending_price'])}")
    print(f" Median ending price    : {money(summary['median_ending_price'])}")
    print(f" 1st percentile         : {money(summary['percentile_1'])}")
    print(f" 5th percentile         : {money(summary['percentile_5'])}")
    print(f" 95th percentile        : {money(summary['percentile_95'])}")
    print(f" 99th percentile        : {money(summary['percentile_99'])}")
    print(f" P(gain > 20%)          : {summary['prob_gain_more_than_20pct']:.2%}")
    print(f" P(drop > 10%)          : {summary['prob_drop_more_than_10pct']:.2%}")
    print("-" * 64)
    print(f" Runtime                : {summary['runtime_seconds']:.3f} s")
    print(f" Memory safety          : {summary['memory_status']}")
    print("=" * 64)


def run(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    market = mc_core.estimate_parameters_from_history(
        args.ticker, years=args.years, s0_override=args.start_price
    )
    config = build_config(
        ticker=args.ticker, s0=market.s0, mu=market.mu, sigma=market.sigma,
        paths=args.paths, horizon=args.horizon, seed=args.seed,
        chunk_size=args.chunk_size,
    )

    result = mc_core.simulate(config)
    summary = summarize(result, market)
    _print_summary(summary)

    written = write_outputs(result, summary, outdir=args.outdir,
                            make_chart=not args.no_chart)
    for kind, path in written.items():
        print(f" [{kind} written] {path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
