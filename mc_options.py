#!/usr/bin/env python3
"""
mc_options.py — option pricing entry point on the v2 engine (Phase 2).

Prices European / Asian / barrier / lookback options with streaming
PathPricers (mc_payoffs.py) and American/Bermudan options with
Longstaff-Schwartz (mc_lsm.py), under GBM or Heston dynamics — the two
models ported to the v2 engine in Phase 1.

The process is always built with **risk-neutral drift mu = r** (the
config's historical/manual drift modes are *not* used — an option price is
only arbitrage-free under the risk-neutral measure; the header of every
run says so). The existing ``simulate()`` risk-analysis path and its
output schema are untouched.

Examples
--------
    # European call, offline (explicit market state):
    python mc_options.py AAPL --option european --strike 200 --maturity 0.5 \
        --r 0.05 --s0 190 --sigma 0.25

    # Down-and-out barrier put under Heston, market data fetched:
    python mc_options.py AAPL --option barrier --put --strike 200 \
        --barrier 150 --barrier-dir down --barrier-knock out \
        --maturity 1.0 --r 0.05 --model heston

    # American put via Longstaff-Schwartz:
    python mc_options.py AAPL --option american --put --strike 200 \
        --maturity 1.0 --r 0.05 --s0 190 --sigma 0.25
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional

import numpy as np

from mc_core import estimate_parameters_from_history, sobol_available
from mc_engine import GBMProcess, HestonProcess, PathGenerator, StochasticProcess
from mc_lsm import longstaff_schwartz_price
from mc_payoffs import (
    AsianArithmeticPricer,
    AsianGeometricPricer,
    BarrierPricer,
    EuropeanPricer,
    LookbackPricer,
)

OPTION_KINDS = ("european", "asian", "asian-geometric", "barrier",
                "lookback", "american")
MODELS = ("gbm", "heston")


def build_process(args, sigma: float, dt: float) -> StochasticProcess:
    """Risk-neutral process: drift is always the risk-free rate."""
    if args.model == "gbm":
        return GBMProcess(args.r, sigma, dt)
    theta = args.heston_theta if args.heston_theta is not None else sigma ** 2
    v0 = args.heston_v0 if args.heston_v0 is not None else sigma ** 2
    return HestonProcess(
        args.r, dt,
        kappa=args.heston_kappa, theta=theta, xi=args.heston_xi,
        rho=args.heston_rho, v0=v0,
    )


def build_pricer(args, maturity: float):
    common = dict(strike=args.strike, maturity=maturity, r=args.r,
                  call=not args.put)
    if args.option == "european":
        return EuropeanPricer(**common)
    if args.option == "asian":
        return AsianArithmeticPricer(**common)
    if args.option == "asian-geometric":
        return AsianGeometricPricer(**common)
    if args.option == "barrier":
        if args.barrier is None:
            raise SystemExit("--barrier is required for --option barrier")
        return BarrierPricer(barrier=args.barrier, direction=args.barrier_dir,
                             knock=args.barrier_knock, **common)
    if args.option == "lookback":
        return LookbackPricer(maturity=maturity, r=args.r, call=not args.put,
                              kind=args.lookback_kind, strike=args.strike)
    raise SystemExit(f"unsupported option kind: {args.option}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Price options on the v2 Monte Carlo engine "
                    "(risk-neutral drift mu = r; GBM or Heston).",
    )
    p.add_argument("ticker", nargs="?", default="ASSET")
    p.add_argument("--option", required=True, choices=OPTION_KINDS)
    p.add_argument("--put", action="store_true", help="price a put (default call)")
    p.add_argument("--strike", type=float, required=True)
    p.add_argument("--maturity", type=float, required=True,
                   help="maturity in years")
    p.add_argument("--r", type=float, required=True,
                   help="continuously compounded risk-free rate")
    p.add_argument("--model", choices=MODELS, default="gbm")
    p.add_argument("--s0", type=float, default=None,
                   help="spot override (else fetched from market history)")
    p.add_argument("--sigma", type=float, default=None,
                   help="annual vol override (else fetched)")
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--steps-per-year", type=int, default=252,
                   help="monitoring/simulation steps per year (default 252)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=50_000)
    p.add_argument("--variance-reduction", choices=("none", "antithetic", "sobol"),
                   default="none")
    # Barrier / lookback specifics
    p.add_argument("--barrier", type=float, default=None)
    p.add_argument("--barrier-dir", choices=("up", "down"), default="down")
    p.add_argument("--barrier-knock", choices=("in", "out"), default="out")
    p.add_argument("--lookback-kind", choices=("fixed", "floating"),
                   default="floating")
    # Heston parameters (theta/v0 default to sigma^2)
    p.add_argument("--heston-kappa", type=float, default=1.5)
    p.add_argument("--heston-theta", type=float, default=None)
    p.add_argument("--heston-xi", type=float, default=0.3)
    p.add_argument("--heston-rho", type=float, default=-0.7)
    p.add_argument("--heston-v0", type=float, default=None)
    # Longstaff-Schwartz specifics
    p.add_argument("--exercise-dates", type=int, default=50,
                   help="Bermudan exercise dates over the option life "
                        "(LSM; 50/year approximates American)")
    p.add_argument("--substeps", type=int, default=1,
                   help="simulation substeps between exercise dates (LSM; "
                        "use >1 for Heston accuracy)")
    p.add_argument("--degree", type=int, default=3,
                   help="LSM polynomial basis degree")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # ---- market state (never synthetic) -------------------------------
    if args.s0 is not None and args.sigma is not None:
        s0, sigma, source = float(args.s0), float(args.sigma), "manual"
    else:
        mkt = estimate_parameters_from_history(args.ticker)
        if mkt.source == "fallback":
            print("ERROR: no market data for "
                  f"{args.ticker!r} and no --s0/--sigma given. Refusing to "
                  "price on synthetic fallback data.", file=sys.stderr)
            return 2
        s0 = float(args.s0) if args.s0 is not None else float(mkt.s0)
        sigma = float(args.sigma) if args.sigma is not None else float(mkt.sigma)
        source = mkt.source

    if args.variance_reduction == "sobol" and not sobol_available():
        print("ERROR: --variance-reduction sobol requires SciPy.",
              file=sys.stderr)
        return 2

    kind = "put" if args.put else "call"
    print(f"=== {args.option.upper()} {kind} on {args.ticker.upper()} "
          f"({args.model.upper()}) ===")
    print(f"S0={s0:.4f}  sigma={sigma:.4%}  (source={source})")
    print(f"strike={args.strike}  maturity={args.maturity}y  r={args.r:.4%}  "
          f"drift=risk-neutral (mu=r)")

    # ---- American / Bermudan via Longstaff-Schwartz ---------------------
    if args.option == "american":
        if args.model == "heston" and args.substeps == 1:
            print("note: consider --substeps > 1 for Heston Euler accuracy")
        res = longstaff_schwartz_price(
            lambda dt: build_process(args, sigma, dt),
            s0=s0, strike=args.strike, maturity=args.maturity, r=args.r,
            call=not args.put, paths=args.paths,
            exercise_dates=args.exercise_dates, substeps=args.substeps,
            degree=args.degree, seed=args.seed, chunk_size=args.chunk_size,
            antithetic=(args.variance_reduction == "antithetic"),
        )
        print(res.summary())
        print(f"stored matrix: {res.stored_matrix_elements:,} elements "
              f"(paths x {res.exercise_dates} exercise dates — documented "
              "LSM exception to full streaming)")
        return 0

    # ---- streaming pricers ----------------------------------------------
    steps = max(1, int(round(args.steps_per_year * args.maturity)))
    dt = args.maturity / steps
    process = build_process(args, sigma, dt)
    gen = PathGenerator(
        process, s0=s0, paths=args.paths, steps=steps,
        chunk_size=args.chunk_size, seed=args.seed,
        antithetic=(args.variance_reduction == "antithetic"),
        sobol=(args.variance_reduction == "sobol"),
    )
    pricer = build_pricer(args, args.maturity)
    gen.run([pricer])
    price, se = pricer.price(), pricer.std_error()
    print(f"steps={steps} (dt={dt:.5f}y)  paths={args.paths:,}  "
          f"VR={args.variance_reduction}")
    print(f"PRICE = {price:.4f}   (naive MC std error {se:.4f})")
    if args.variance_reduction == "antithetic":
        print("note: antithetic pairing makes the true std error smaller "
              "than the naive estimate above")
    if args.variance_reduction == "sobol":
        print("note: Sobol is deterministic-ish; the naive std error is not "
              "a valid QMC error bound — use replications")
    if args.option == "barrier":
        print("note: discrete monitoring — knock-outs price above, knock-ins "
              "below, their continuous-barrier closed forms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
