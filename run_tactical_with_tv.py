#!/usr/bin/env python3
"""
Example: short-horizon tactical Monte Carlo driven by TradingView data.

What this script does
---------------------
1. Optionally writes a *demo* signal file (so you can try without TradingView).
2. Loads the latest signal from the Phase 3 webhook bridge folder.
3. Runs a 5-day tactical simulation that:
      - uses TV ticker & price
      - aligns long/short to TV trend
      - scales volatility from TV momentum (RSI)
4. Prints a clear summary showing whether TradingView data was used.

Usage
-----
# Offline demo (creates a fake signal, no webhook needed):
    python run_tactical_with_tv.py --demo

# Real bridge file (after tv_webhook_bridge received an alert):
    python run_tactical_with_tv.py

# Custom paths / size:
    python run_tactical_with_tv.py --demo --paths 10000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a tactical MC simulation using TradingView bridge data.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Write a demo signal into tv_data/ before running (offline).",
    )
    p.add_argument("--ticker", default="AAPL", help="Demo ticker (default AAPL).")
    p.add_argument("--price", type=float, default=190.0, help="Demo price.")
    p.add_argument(
        "--trend",
        default="bullish",
        choices=["bullish", "bearish"],
        help="Demo trend direction.",
    )
    p.add_argument(
        "--momentum",
        type=float,
        default=62.0,
        help="Demo RSI-like momentum 0–100 (default 62).",
    )
    p.add_argument("--paths", type=int, default=15_000, help="Monte Carlo paths.")
    p.add_argument("--horizon", type=int, default=5, help="Trading days ahead.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop", type=float, default=0.02, help="Stop-loss fraction.")
    p.add_argument("--tp", type=float, default=0.03, help="Take-profit fraction.")
    p.add_argument(
        "--data-dir",
        default="tv_data",
        help="Folder with latest_signal.json (default tv_data).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=0.25,
        help="Base annual vol before momentum scaling (default 0.25).",
    )
    p.add_argument(
        "--require-signal",
        action="store_true",
        help="Fail if no TradingView signal file is present.",
    )
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    from tactical_config import TradingRule, preset_5_day, preset_10_day
    from tactical_simulator import run_tactical_simulation
    from tv_integration import write_demo_signal

    data_dir = Path(args.data_dir)
    if args.demo:
        path = write_demo_signal(
            data_dir / "latest_signal.json",
            ticker=args.ticker,
            price=args.price,
            trend=args.trend,
            momentum=args.momentum,
        )
        print(f"[demo] Wrote sample TradingView signal → {path.resolve()}\n")

    if args.horizon <= 5:
        cfg = preset_5_day(
            args.ticker,
            paths=int(args.paths),
            seed=int(args.seed),
            annual_volatility=float(args.sigma),
            annual_drift=0.0,
            starting_price=float(args.price) if args.demo else None,
        )
    else:
        from dataclasses import replace

        cfg = preset_10_day(
            args.ticker,
            paths=int(args.paths),
            seed=int(args.seed),
            annual_volatility=float(args.sigma),
            annual_drift=0.0,
        )
        cfg = replace(cfg, horizon_days=int(args.horizon))

    rule = TradingRule(
        name=f"TV-integrated {args.horizon}d",
        entry_condition="Enter with TV trend alignment",
        exit_condition="Stop / take-profit / max hold",
        stop_loss_pct=float(args.stop),
        take_profit_pct=float(args.tp),
        max_holding_days=int(args.horizon),
        side="long",  # may be overwritten by TV trend
    )
    cfg = cfg.with_rule(rule)

    print("Running tactical simulation with use_tradingview=True …\n")
    result = run_tactical_simulation(
        cfg,
        use_tradingview=True,
        tv_data_dir=data_dir,
        tv_align_side=True,
        tv_scale_vol=True,
        tv_scale_jumps=True,
        tv_require_signal=bool(args.require_signal),
    )
    print(result.summary_text())
    print()
    print(
        "Tip: keep `python tv_webhook_bridge.py --secret …` running and drop "
        "--demo to use a real TradingView alert file."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
