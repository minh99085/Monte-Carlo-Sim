"""
weekly_decision.py — honest weekly trade decision layer on top of Monte-Carlo-Sim.

What this fixes relative to using the raw simulator for weekly trading
----------------------------------------------------------------------
1. DRIFT IS AN EXPLICIT INPUT, NEVER AN ESTIMATE.
   A 3-year historical mu has a standard error larger than any plausible
   equity edge, so this tool never "estimates" drift and calls it a forecast.
   You supply your expected *weekly* return from your own signal
   (``--signal-weekly``), or it runs at zero drift and tells you the
   breakeven edge the trade structure requires.

2. VOLATILITY IS CURRENT, NOT 3-YEAR.
   Uses RiskMetrics EWMA (lambda=0.94) on recent daily returns, which reacts
   to the current vol regime. Falls back to a supplied --sigma when offline.

3. THE STOP IS CALIBRATED TO NOISE, NOT A MAGIC 2%.
   Stops are expressed as multiples of the *weekly* sigma. The tool sweeps a
   stop/TP grid at zero drift and reports P(stopped out by pure noise) for
   each, so you never pick a stop that random diffusion will hit 40% of the
   time.

4. BREAKEVEN EDGE, NOT FAKE P(PROFIT).
   For the chosen structure it bisects over drift to find the annualized mu
   at which expected P&L (after costs, stops, TP) crosses zero. Your decision
   becomes: "does my signal credibly deliver more than X% annualized over
   the next 5 days?" — a falsifiable question.

5. NON-OVERLAPPING HISTORICAL VALIDATION.
   The repo's rolling backtest slides windows with step=1 (adjacent 5-day
   windows share 4 days -> heavily autocorrelated, n_windows overstates the
   sample). This tool uses step = horizon so windows are independent.

6. SIZING VIA CAPPED FRACTIONAL KELLY.
   From the simulated P&L% distribution under YOUR drift assumption, with a
   hard cap, and sized to a max account risk per trade.

Usage
-----
    # No signal: shows noise stop-out rates + breakeven edge (no trade verdict
    # is possible without a signal — and it will say so).
    python weekly_decision.py AAPL

    # With your signal's expected weekly return (e.g. +0.6% over 5 days):
    python weekly_decision.py AAPL --signal-weekly 0.006

    # Offline / manual:
    python weekly_decision.py AAPL --s0 200 --sigma 0.28 --signal-weekly 0.006
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from mc_core import TRADING_DAYS_PER_YEAR, estimate_parameters_from_history
from tactical_config import TacticalConfig, TradingRule
from tactical_simulator import (
    run_historical_rule_backtest,
    run_tactical_simulation,
)

WEEK_DAYS = 5
EWMA_LAMBDA = 0.94
KELLY_CAP = 0.25          # never size above quarter-Kelly
MAX_RISK_PER_TRADE = 0.01  # cap: 1% of account at the stop
DEFAULT_COST = 0.0005      # 5 bps round-trip per side basis (repo convention)


# ---------------------------------------------------------------------------
# Vol estimation (EWMA — reacts to the current regime)
# ---------------------------------------------------------------------------

def ewma_annual_sigma(daily_log_returns: np.ndarray,
                      lam: float = EWMA_LAMBDA) -> float:
    r = np.asarray(daily_log_returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 20:
        raise ValueError("need >= 20 daily returns for EWMA vol")
    var = float(np.var(r[:20]))
    for x in r[20:]:
        var = lam * var + (1.0 - lam) * float(x) * float(x)
    return math.sqrt(var * TRADING_DAYS_PER_YEAR)


# ---------------------------------------------------------------------------
# Simulation wrapper
# ---------------------------------------------------------------------------

def _base_config(ticker: str, s0: float, sigma: float, mu_annual: float,
                 rule: TradingRule, paths: int, cost: float,
                 seed: int) -> TacticalConfig:
    return TacticalConfig(
        ticker=ticker,
        horizon_days=WEEK_DAYS,
        paths=paths,
        seed=seed,
        starting_price=float(s0),
        annual_volatility=float(sigma),
        annual_drift=float(mu_annual),
        transaction_cost=float(cost),
        sample_paths=0,
        trading_rule=rule,
    ).validate()


def run_structure(ticker: str, s0: float, sigma: float, mu_annual: float,
                  rule: TradingRule, paths: int, cost: float,
                  seed: int) -> dict:
    cfg = _base_config(ticker, s0, sigma, mu_annual, rule, paths, cost, seed)
    res = run_tactical_simulation(cfg)
    st = dict(res.stats)
    st["pnl_pct"] = np.asarray(res.pnl_pct, dtype=float)
    return st


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

@dataclass
class StructureRow:
    stop_mult: float           # stop as multiple of weekly sigma
    tp_mult: Optional[float]   # TP as multiple of weekly sigma (None = no TP)
    stop_pct: float
    tp_pct: Optional[float]
    noise_stop_rate: float     # P(stop hit | zero drift)
    noise_avg_pnl_pct: float   # expectancy at zero drift (structure drag)
    breakeven_mu: float        # annualized drift where expectancy = 0


def weekly_sigma(annual_sigma: float) -> float:
    return annual_sigma * math.sqrt(WEEK_DAYS / TRADING_DAYS_PER_YEAR)


def make_rule(stop_pct: float, tp_pct: Optional[float],
              side: str = "long") -> TradingRule:
    return TradingRule(
        side=side,
        entry_day=0,
        stop_loss_pct=float(stop_pct),
        take_profit_pct=(float(tp_pct) if tp_pct is not None else None),
        max_holding_days=WEEK_DAYS,
        allow_reentry=False,
    )


def breakeven_drift(ticker: str, s0: float, sigma: float, rule: TradingRule,
                    paths: int, cost: float, seed: int,
                    lo: float = -1.0, hi: float = 3.0,
                    tol: float = 0.005, max_iter: int = 24) -> float:
    """Bisect annualized mu until mean pnl_pct crosses zero (fixed seed ->
    common random numbers, so the objective is monotone and smooth)."""

    def expectancy(mu: float) -> float:
        st = run_structure(ticker, s0, sigma, mu, rule, paths, cost, seed)
        return float(np.mean(st["pnl_pct"]))

    f_lo, f_hi = expectancy(lo), expectancy(hi)
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            return mid
        if expectancy(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sweep_structures(ticker: str, s0: float, sigma: float, paths: int,
                     cost: float, seed: int,
                     stop_mults: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
                     tp_mults: Tuple[Optional[float], ...] = (None, 1.0, 2.0),
                     ) -> List[StructureRow]:
    wsig = weekly_sigma(sigma)
    rows: List[StructureRow] = []
    for sm in stop_mults:
        for tm in tp_mults:
            stop_pct = min(0.99, sm * wsig)
            tp_pct = (tm * wsig) if tm is not None else None
            rule = make_rule(stop_pct, tp_pct)
            zero = run_structure(ticker, s0, sigma, 0.0, rule,
                                 paths, cost, seed)
            be = breakeven_drift(ticker, s0, sigma, rule,
                                 max(paths // 2, 10_000), cost, seed)
            rows.append(StructureRow(
                stop_mult=sm, tp_mult=tm,
                stop_pct=stop_pct, tp_pct=tp_pct,
                noise_stop_rate=float(zero.get("stop_hit_rate", float("nan"))),
                noise_avg_pnl_pct=float(np.mean(zero["pnl_pct"])),
                breakeven_mu=be,
            ))
    return rows


def kelly_fraction(pnl_pct: np.ndarray) -> float:
    """Discrete Kelly approximation f* ~ mean/variance of per-trade return,
    capped. Returns 0 if expectancy <= 0."""
    m = float(np.mean(pnl_pct))
    v = float(np.var(pnl_pct))
    if m <= 0 or v <= 0:
        return 0.0
    return min(m / v, 1.0) * KELLY_CAP


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_pct(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "  n/a"
    return f"{100.0 * x:.{nd}f}%"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("ticker")
    ap.add_argument("--signal-weekly", type=float, default=None,
                    help="YOUR expected return over the 5-day hold, e.g. 0.006 "
                         "= +0.6%%. Comes from your signal, not this tool.")
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--s0", type=float, default=None,
                    help="starting price override (offline use)")
    ap.add_argument("--sigma", type=float, default=None,
                    help="annualized vol override (offline use)")
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST,
                    help="proportional cost per side (default 5 bps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-mult", type=float, default=1.5,
                    help="chosen stop as multiple of weekly sigma")
    ap.add_argument("--tp-mult", type=float, default=None,
                    help="optional TP as multiple of weekly sigma")
    ap.add_argument("--account", type=float, default=100_000.0,
                    help="account size for position sizing")
    ap.add_argument("--history-check", action="store_true",
                    help="also run a NON-overlapping historical rule backtest")
    args = ap.parse_args(argv)

    # ---- market state -----------------------------------------------------
    hist_returns = None
    if args.s0 is not None and args.sigma is not None:
        s0, sigma, src = float(args.s0), float(args.sigma), "manual"
    else:
        mkt = estimate_parameters_from_history(args.ticker)
        src = mkt.source
        if src == "fallback" and (args.s0 is None or args.sigma is None):
            print("ERROR: no market data available and no --s0/--sigma "
                  "provided. Refusing to invent parameters.", file=sys.stderr)
            return 2
        s0 = float(args.s0) if args.s0 is not None else float(mkt.s0)
        hist_returns = np.asarray(mkt.daily_log_returns, dtype=float)
        if args.sigma is not None:
            sigma = float(args.sigma)
        else:
            sigma = ewma_annual_sigma(hist_returns)  # NOT the 3y sigma
            src = f"{src} (EWMA λ={EWMA_LAMBDA})"

    wsig = weekly_sigma(sigma)
    print(f"\n=== WEEKLY TRADE CHECK: {args.ticker.upper()} ({args.side}) ===")
    print(f"S0={s0:.2f}  annual σ={fmt_pct(sigma,1)}  "
          f"weekly σ={fmt_pct(wsig,2)}  source={src}")
    print(f"cost={args.cost*1e4:.0f} bps/side  horizon={WEEK_DAYS}d  "
          f"paths={args.paths:,}\n")

    # ---- structure sweep at ZERO drift ------------------------------------
    print("STRUCTURE SWEEP (zero drift — what pure noise does to each setup)")
    print("  stop     TP      P(noise stop-out)  drag/trade   breakeven μ(ann)")
    rows = sweep_structures(args.ticker, s0, sigma, args.paths,
                            args.cost, args.seed)
    for r in rows:
        tp_s = f"{r.tp_mult:.1f}σw" if r.tp_mult is not None else "  — "
        print(f"  {r.stop_mult:.1f}σw ({fmt_pct(r.stop_pct)})  {tp_s:6s}"
              f"   {fmt_pct(r.noise_stop_rate,1):>8s}"
              f"      {fmt_pct(r.noise_avg_pnl_pct):>8s}"
              f"      {fmt_pct(r.breakeven_mu,1):>8s}")

    # ---- chosen structure --------------------------------------------------
    stop_pct = min(0.99, args.stop_mult * wsig)
    tp_pct = (args.tp_mult * wsig) if args.tp_mult is not None else None
    rule = make_rule(stop_pct, tp_pct, side=args.side)
    be = breakeven_drift(args.ticker, s0, sigma, rule,
                         max(args.paths // 2, 10_000), args.cost, args.seed)
    be_weekly = be * WEEK_DAYS / TRADING_DAYS_PER_YEAR

    print(f"\nCHOSEN STRUCTURE: stop={fmt_pct(stop_pct)} "
          f"({args.stop_mult:.1f}σw)"
          + (f", TP={fmt_pct(tp_pct)}" if tp_pct else ", no TP")
          + f", max hold {WEEK_DAYS}d")
    print(f"Breakeven edge: μ={fmt_pct(be,1)} annualized "
          f"≈ {fmt_pct(be_weekly)} over the week")

    # ---- optional independent historical validation ------------------------
    if args.history_check and hist_returns is not None and hist_returns.size:
        prices = s0 * np.exp(np.concatenate([[0.0], np.cumsum(hist_returns)]))
        hb = run_historical_rule_backtest(
            prices, rule, horizon_days=WEEK_DAYS, cost=args.cost,
            step=WEEK_DAYS,  # NON-overlapping windows (repo default step=1
        )                    # double-counts: adjacent windows share 4 days)
        print(f"\nHISTORICAL CHECK (non-overlapping, n={hb.n_windows} weeks)")
        print(f"  P(profit)={fmt_pct(hb.stats['prob_profit'],1)}  "
              f"avg={fmt_pct(hb.stats['avg_pnl']/s0 if s0 else float('nan'))} "
              f" stop-out rate={fmt_pct(hb.stats['stop_hit_rate'],1)}")
        print("  NOTE: in-sample description of the past, not a forecast.")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 60)
    if args.signal_weekly is None:
        print("VERDICT: NO TRADE — no signal supplied.")
        print("This tool cannot manufacture edge. Supply your signal's")
        print(f"expected weekly return via --signal-weekly. It must exceed")
        print(f"{fmt_pct(be_weekly)} (breakeven for this structure) to be +EV.")
        return 0

    mu_annual = (args.signal_weekly
                 * TRADING_DAYS_PER_YEAR / WEEK_DAYS)
    if args.side == "short":
        mu_annual = -mu_annual  # signal is edge in trade direction
    st = run_structure(args.ticker, s0, sigma, mu_annual, rule,
                       args.paths, args.cost, args.seed)
    pnl = st["pnl_pct"]
    exp_ret = float(np.mean(pnl))
    p_win = float(np.mean(pnl > 0))
    p5, p95 = np.percentile(pnl, [5, 95])

    print(f"UNDER YOUR SIGNAL ({fmt_pct(args.signal_weekly)} weekly, "
          f"{fmt_pct(mu_annual,0)} annualized):")
    print(f"  expectancy/trade = {fmt_pct(exp_ret)}   "
          f"P(win)={fmt_pct(p_win,1)}   5–95% = "
          f"[{fmt_pct(p5)}, {fmt_pct(p95)}]")
    print(f"  stop-out rate    = {fmt_pct(st.get('stop_hit_rate'),1)}")

    margin = abs(args.signal_weekly) - be_weekly
    if margin <= 0:
        print(f"\nVERDICT: NO TRADE. Your stated edge {fmt_pct(args.signal_weekly)} "
              f"< breakeven {fmt_pct(be_weekly)}.")
        return 0

    f_kelly = kelly_fraction(pnl)
    notional_kelly = f_kelly * args.account
    notional_riskcap = (MAX_RISK_PER_TRADE * args.account) / stop_pct
    notional = min(notional_kelly, notional_riskcap)
    shares = int(notional // s0)
    print(f"\nVERDICT: TRADE (edge margin {fmt_pct(margin)} over breakeven), "
          f"IF you trust the signal.")
    print(f"  size: {KELLY_CAP:.0%}-Kelly={fmt_pct(f_kelly,1)} of account "
          f"-> ${notional_kelly:,.0f}; risk cap "
          f"({MAX_RISK_PER_TRADE:.0%} at stop) -> ${notional_riskcap:,.0f}")
    print(f"  => {shares:,} shares (~${shares * s0:,.0f}) "
          f"stop {fmt_pct(stop_pct)} "
          + (f"TP {fmt_pct(tp_pct)} " if tp_pct else "")
          + f"time-exit day {WEEK_DAYS}")
    print("\nReminder: expectancy above is conditional on YOUR drift input.")
    print("The simulator prices the structure; it does not validate the signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
