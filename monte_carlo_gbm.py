#!/usr/bin/env python3
"""Command line Monte Carlo GBM simulator.

Runs a memory-safe, chunked Geometric Brownian Motion Monte Carlo simulation for
a single asset and prints a risk summary (expected/median value, probability of
profit/loss, Value at Risk and Expected Shortfall, percentile table, runtime and
memory-safety status).

Examples
--------
    python monte_carlo_gbm.py AAPL --paths 1000 --horizon 10 --no-chart
    python monte_carlo_gbm.py MSFT --paths 100000 --horizon 252 --years 5
    python monte_carlo_gbm.py TSLA --paths 1000000 --chunk-size 50000 --seed 7

The heavy lifting lives in :mod:`mc_core`, which is shared with the Streamlit GUI
(``app.py``).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import mc_core


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="monte_carlo_gbm.py",
        description="Memory-safe, chunked GBM Monte Carlo simulation (CPU-first).",
    )
    p.add_argument("ticker", nargs="?", default="ASSET",
                   help="Ticker symbol used to fetch history (default: ASSET).")
    p.add_argument("--paths", type=int, default=100_000,
                   help="Number of simulated paths (default: 100000).")
    p.add_argument("--horizon", type=int, default=252,
                   help="Number of forward trading-day steps (default: 252).")
    p.add_argument("--years", type=float, default=3.0,
                   help="Years of history used to estimate mu/sigma (default: 3).")
    p.add_argument("--start-price", "--s0", dest="start_price", type=float, default=None,
                   help="Override the starting price S0.")
    p.add_argument("--mu", type=float, default=None,
                   help="Override the annualized drift mu.")
    p.add_argument("--sigma", type=float, default=None,
                   help="Override the annualized volatility sigma.")
    p.add_argument("--chunk-size", type=int, default=mc_core.DEFAULT_SERIOUS_CHUNK,
                   help="Paths simulated per chunk (default: 50000).")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--cost", type=float, default=0.0,
                   help="Proportional transaction cost / slippage (e.g. 0.001).")
    p.add_argument("--sample-paths", type=int, default=50,
                   help="Number of full sample trajectories retained (default: 50).")
    p.add_argument("--export-csv", default=None,
                   help="Write a CSV summary to this path.")
    p.add_argument("--export-json", default=None,
                   help="Write a JSON report to this path.")
    p.add_argument("--no-chart", action="store_true",
                   help="Do not render or save any charts.")
    p.add_argument("--chart-path", default=None,
                   help="Save the sample-path/histogram chart to this PNG path.")
    return p


def _resolve_market(args: argparse.Namespace) -> mc_core.MarketParameters:
    market = mc_core.estimate_parameters_from_history(
        args.ticker, years=args.years, s0_override=args.start_price
    )
    # Apply explicit overrides on top of fetched/fallback values.
    mu = args.mu if args.mu is not None else market.mu
    sigma = args.sigma if args.sigma is not None else market.sigma
    return mc_core.MarketParameters(
        s0=market.s0, mu=mu, sigma=sigma, source=market.source, note=market.note
    )


def _print_summary(result: mc_core.SimulationResult, market: mc_core.MarketParameters) -> None:
    cfg = result.config
    s = result.stats

    def money(x: float) -> str:
        return f"{x:,.2f}"

    print("=" * 64)
    print(f" Monte Carlo GBM simulation: {cfg.ticker}")
    print("=" * 64)
    print(f" Data source       : {market.source}")
    if market.note:
        print(f"   note            : {market.note}")
    print(f" Starting price S0 : {money(cfg.s0)}")
    print(f" Drift mu (annual) : {cfg.mu:.4%}")
    print(f" Vol sigma (annual): {cfg.sigma:.4%}")
    print(f" Paths             : {cfg.paths:,}")
    print(f" Horizon (steps)   : {cfg.horizon}")
    print(f" Chunk size        : {cfg.chunk_size:,}")
    print(f" Seed              : {cfg.seed}")
    print(f" Transaction cost  : {cfg.cost:.4%}")
    print("-" * 64)
    print(f" Expected ending value : {money(s['expected_value'])}  "
          f"({s['expected_return']:+.2%})")
    print(f" Median ending value   : {money(s['median_value'])}")
    print(f" Std dev               : {money(s['std_value'])}")
    print(f" Min / Max             : {money(s['min_value'])} / {money(s['max_value'])}")
    print(f" Probability of profit : {s['prob_profit']:.2%}")
    print(f" Probability of loss   : {s['prob_loss']:.2%}")
    print("-" * 64)
    print(" Value at Risk (loss vs S0)")
    for level in mc_core.RISK_LEVELS:
        key = mc_core._level_key(level)
        v = s["var"][key]
        print(f"   VaR {key:>5}%  : {money(v['value'])}  ({v['pct']:.2%})")
    print(" Expected Shortfall (CVaR)")
    for level in mc_core.RISK_LEVELS:
        key = mc_core._level_key(level)
        e = s["expected_shortfall"][key]
        print(f"   ES  {key:>5}%  : {money(e['value'])}  ({e['pct']:.2%})")
    print("-" * 64)
    print(" Percentile table (ending value)")
    for p, v in s["percentiles"].items():
        print(f"   p{p:>3} : {money(v)}")
    print("-" * 64)
    print(f" Runtime           : {result.runtime_seconds:.3f} s "
          f"({s.get('paths_per_second', 0):,.0f} paths/s)")
    print(f" Memory safety     : {result.memory.status()}")
    print("=" * 64)


def _maybe_chart(result: mc_core.SimulationResult, args: argparse.Namespace) -> None:
    if args.no_chart and not args.chart_path:
        return
    try:
        import matplotlib
        if args.chart_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[chart skipped] matplotlib unavailable: {exc}", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    traj = result.sample_trajectories
    for i in range(traj.shape[0]):
        ax1.plot(traj[i], linewidth=0.7, alpha=0.7)
    ax1.set_title(f"{result.config.ticker}: sample paths "
                  f"({traj.shape[0]} of {result.config.paths:,})")
    ax1.set_xlabel("step")
    ax1.set_ylabel("price")

    ax2.hist(result.final_values, bins=80, color="#3b7dd8", alpha=0.85)
    ax2.axvline(result.config.s0, color="black", linestyle="--", linewidth=1,
                label="S0")
    ax2.set_title("Ending-value distribution")
    ax2.set_xlabel("ending value")
    ax2.set_ylabel("frequency")
    ax2.legend()
    fig.tight_layout()

    if args.chart_path:
        fig.savefig(args.chart_path, dpi=110)
        print(f"[chart saved] {args.chart_path}")
    else:
        plt.show()
    plt.close(fig)


def run(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    market = _resolve_market(args)

    config = mc_core.SimulationConfig(
        ticker=args.ticker,
        s0=market.s0,
        paths=args.paths,
        horizon=args.horizon,
        mu=market.mu,
        sigma=market.sigma,
        chunk_size=args.chunk_size,
        seed=args.seed,
        cost=args.cost,
        sample_paths=args.sample_paths,
    )

    result = mc_core.simulate(config)
    _print_summary(result, market)

    if args.export_csv:
        mc_core.write_csv(result, args.export_csv)
        print(f"[csv exported] {args.export_csv}")
    if args.export_json:
        mc_core.write_json(result, args.export_json, market)
        print(f"[json exported] {args.export_json}")

    _maybe_chart(result, args)
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
