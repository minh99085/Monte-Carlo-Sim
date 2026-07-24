#!/usr/bin/env python3
"""
run_weekly_from_tv.py — the single production entrypoint for the
TradingView → calibrated-drift → Monte Carlo weekly decision pipeline.

Flow
----
1. Read the latest bridge signal (``tv_data/latest_signal.json``).
2. Freshness check: signals older than ``--max-age-hours`` (default 30) are
   treated as **no signal**.
3. Calibration lookup: the signal's trend/RSI bucket maps to a shrunken
   conditional drift (``signal_calibration.py``); missing/stale calibration or
   an undefined bucket → drift 0.0, reason recorded.
4. Volatility: EWMA(λ=0.94) realized vol from recent history — never the
   RSI multiplier, never a 3-year flat sigma.
5. Decision layer (reuses ``weekly_decision.py``): breakeven drift by
   bisection with common random numbers, stop in weekly-sigma multiples with
   a zero-drift noise-stop-out check, quarter-Kelly sizing capped at 1%
   account risk at the stop.
6. Verdict block printed with the drift source labeled (bucket, t, n_eff) —
   no naked probabilities — and written to
   ``outputs/verdicts/{ts}_{ticker}.json``; every verdict is also appended to
   the trade log (``outcome_tracker.log_verdict``).

Hard rules
----------
* If market data resolution returns ``source == "fallback"`` (synthetic
  data), exit with an error. Never a verdict from invented data.
* If the outcome-tracker kill-switch is tripped (trailing 20 settled TRADE
  verdicts with negative mean realized P&L), TRADE verdicts are refused
  until ``--override-killswitch`` is passed.

Usage
-----
    python run_weekly_from_tv.py                  # latest bridge signal
    python run_weekly_from_tv.py --demo           # offline demo signal
    python run_weekly_from_tv.py --s0 190 --sigma 0.25   # offline market
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mc_core import TRADING_DAYS_PER_YEAR, estimate_parameters_from_history
from tv_integration import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_MAX_SIGNAL_AGE_HOURS,
    DEFAULT_TV_DATA_DIR,
    DriftEstimate,
    load_tradingview_signal,
    signal_age_hours,
    signal_to_drift,
    trend_to_side,
    write_demo_signal,
    _as_float,
    _as_str,
)
from weekly_decision import (
    EWMA_LAMBDA,
    KELLY_CAP,
    MAX_RISK_PER_TRADE,
    ewma_annual_sigma,
    kelly_fraction,
    make_rule,
    run_structure,
    weekly_sigma,
)
import outcome_tracker

WEEK_DAYS = 5
DEFAULT_COST = 0.0005  # 5 bps per side (Phase D4: parameter, not hardcoded 10)
DEFAULT_VERDICT_DIR = Path("outputs") / "verdicts"


class PipelineError(Exception):
    """Base error: the pipeline cannot produce a verdict."""

    exit_code = 1


class NoSignalError(PipelineError):
    exit_code = 3


class FallbackDataError(PipelineError):
    """Market data resolution returned synthetic fallback data."""

    exit_code = 2


# ---------------------------------------------------------------------------
# Decision helpers
# ---------------------------------------------------------------------------


def breakeven_edge(
    ticker: str,
    s0: float,
    sigma: float,
    rule,
    side: str,
    paths: int,
    cost: float,
    seed: int,
    lo: float = 0.0,
    hi: float = 3.0,
    tol: float = 0.005,
    max_iter: int = 24,
) -> float:
    """Annualized drift *edge in the trade direction* at which the structure's
    expectancy crosses zero (common random numbers via fixed seed).

    For a long rule the simulated mu equals the edge; for a short rule the
    simulated mu is the negated edge (the underlying must fall).
    """
    sign = -1.0 if side == "short" else 1.0

    def expectancy(edge: float) -> float:
        st = run_structure(ticker, s0, sigma, sign * edge, rule, paths,
                           cost, seed)
        return float(np.mean(st["pnl_pct"]))

    if expectancy(lo) > 0:
        return lo
    if expectancy(hi) < 0:
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


def fmt_pct(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100.0 * x:.{nd}f}%"


def momentum_lens_check(
    ticker: str,
    daily_log_returns: Optional[np.ndarray],
    side: str,
    horizon_days: int,
    calibration_dir: Path | str,
) -> Dict[str, Any]:
    """Second, independent witness: the 12-1 time-series-momentum lens.

    Computes today's momentum state from the resolved price history and looks
    it up in the momentum calibration table. ``active`` is False (with the
    reason in ``note``) when the check cannot run — missing table, manual
    market override, short history — so deployments without momentum tables
    keep working until the next recalibration. When active, ``agrees`` is
    True only for a verified edge in the trade direction.
    """
    from signal_calibration import (
        FEATURE_SET_MOMENTUM,
        MOM_LOOKBACK_DAYS,
        CalibrationError,
        CalibrationStaleError,
        load_calibration,
        momentum_12_1,
        momentum_bucket,
    )

    lens: Dict[str, Any] = {
        "active": False, "bucket": None, "mom_12_1": None,
        "shrunk_mu_weekly": None, "shrunk_mu_annual": None,
        "t_stat": None, "n_eff": None, "direction": "none",
        "agrees": None, "note": "",
    }
    if daily_log_returns is None:
        lens["note"] = ("no price history (manual market override) — "
                        "momentum lens inactive")
        return lens
    r = np.asarray(daily_log_returns, dtype=float).ravel()
    if r.size < MOM_LOOKBACK_DAYS:
        lens["note"] = (f"history too short for 12-1 momentum "
                        f"({r.size} < {MOM_LOOKBACK_DAYS} returns) — "
                        "momentum lens inactive")
        return lens
    # Momentum uses price *ratios* only, so a relative path reconstructed
    # from returns is exact.
    rel_prices = np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    mom = float(momentum_12_1(rel_prices)[-1])
    lens["mom_12_1"] = mom
    try:
        bucket = momentum_bucket(mom)
    except ValueError as exc:
        lens["note"] = f"undefined momentum bucket: {exc}"
        return lens
    lens["bucket"] = bucket
    try:
        table = load_calibration(ticker, horizon_days, calibration_dir,
                                 feature_set=FEATURE_SET_MOMENTUM)
    except FileNotFoundError:
        lens["note"] = (f"no momentum calibration table for {ticker.upper()} "
                        f"({horizon_days}d) — run signal_calibration.py to "
                        "activate the agreement filter")
        return lens
    except CalibrationStaleError as exc:
        lens["note"] = f"momentum calibration too stale: {exc}"
        return lens
    except CalibrationError as exc:
        lens["note"] = f"momentum calibration unreadable: {exc}"
        return lens

    stats = table.get(bucket)
    if stats is None or stats.n < 2:
        lens.update(active=True, agrees=False,
                    note=f"momentum bucket {bucket} has no usable samples — "
                         "second witness cannot confirm")
        return lens
    mu_w = float(stats.shrunk_mu_weekly)
    direction = "long" if mu_w > 0 else ("short" if mu_w < 0 else "none")
    lens.update(
        active=True,
        shrunk_mu_weekly=mu_w,
        shrunk_mu_annual=float(stats.shrunk_mu_annual),
        t_stat=float(stats.t_stat),
        n_eff=float(stats.n_eff),
        direction=direction,
    )
    if mu_w == 0.0:
        lens["agrees"] = False
        lens["note"] = (f"momentum lens sees no verified edge in {bucket} "
                        f"(t={stats.t_stat:.2f}, n_eff={stats.n_eff:.0f}) — "
                        "second witness says noise")
    elif direction != side:
        lens["agrees"] = False
        lens["note"] = (f"momentum lens points {direction} "
                        f"({stats.shrunk_mu_annual:+.1%}/yr in {bucket}) "
                        f"but the trade side is {side}")
    else:
        lens["agrees"] = True
        lens["note"] = (f"momentum lens agrees: {direction} "
                        f"{stats.shrunk_mu_annual:+.1%}/yr in {bucket} "
                        f"[t={stats.t_stat:.2f}, n_eff={stats.n_eff:.0f}]")
    return lens


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineSettings:
    data_dir: Path = DEFAULT_TV_DATA_DIR
    signal_path: Optional[Path] = None
    calibration_dir: Path = Path(DEFAULT_CALIBRATION_DIR)
    horizon_days: int = WEEK_DAYS
    max_age_hours: float = DEFAULT_MAX_SIGNAL_AGE_HOURS
    paths: int = 100_000
    be_paths: Optional[int] = None       # default: max(paths // 2, 10_000)
    seed: int = 42
    cost: float = DEFAULT_COST           # per side
    stop_mult: float = 1.5               # stop as multiple of weekly sigma
    tp_mult: Optional[float] = None
    account: float = 100_000.0
    s0: Optional[float] = None           # manual market override
    sigma: Optional[float] = None
    years_history: float = 3.0
    trade_log: Path = field(
        default_factory=lambda: Path(outcome_tracker.DEFAULT_TRADE_LOG))
    verdict_dir: Path = field(default_factory=lambda: DEFAULT_VERDICT_DIR)
    override_killswitch: bool = False
    write_files: bool = True
    # Zero-drift noise-stop-out sweep across stop multiples (informational;
    # disable for speed in bulk runs/tests).
    noise_sweep: bool = True
    sweep_stop_mults: tuple = (0.5, 1.0, 1.5, 2.0)
    # Cost stress: a TRADE must also beat the breakeven computed at
    # cost_stress_mult × the assumed per-side cost (<=1 disables).
    cost_stress_mult: float = 2.0
    # Agreement filter: require the independent 12-1 momentum lens to show a
    # verified same-direction edge before TRADE (inactive with a recorded
    # reason when no momentum table / history is available).
    agreement_filter: bool = True


def _resolve_market(settings: PipelineSettings, ticker: str,
                    signal_price: Optional[float]) -> Dict[str, Any]:
    """Resolve s0 / sigma / history. Hard-fails on synthetic fallback data."""
    if settings.s0 is not None and settings.sigma is not None:
        s0 = float(settings.s0)
        return {
            "s0": s0,
            "sigma": float(settings.sigma),
            "source": "manual",
            "sigma_method": "manual override",
            "daily_log_returns": None,
        }

    mkt = estimate_parameters_from_history(ticker, years=settings.years_history)
    if mkt.source == "fallback":
        raise FallbackDataError(
            f"Market data for {ticker} resolved to source='fallback' "
            "(synthetic). Refusing to produce a verdict from invented data. "
            "Provide --s0/--sigma or restore network access."
        )
    returns = np.asarray(mkt.daily_log_returns, dtype=float)
    if settings.sigma is not None:
        sigma = float(settings.sigma)
        sigma_method = "manual override"
    else:
        sigma = ewma_annual_sigma(returns)
        sigma_method = f"EWMA(λ={EWMA_LAMBDA}) realized vol"
    if settings.s0 is not None:
        s0 = float(settings.s0)
    elif signal_price:
        s0 = float(signal_price)
    else:
        s0 = float(mkt.s0)
    return {
        "s0": s0,
        "sigma": sigma,
        "source": mkt.source,
        "sigma_method": sigma_method,
        "daily_log_returns": returns,
    }


def run_pipeline(settings: PipelineSettings) -> Dict[str, Any]:
    """Run the full decision pipeline; returns the verdict dict.

    Raises :class:`NoSignalError` when there is no fresh signal and
    :class:`FallbackDataError` on synthetic market data.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # ---- 1-2. signal + freshness ---------------------------------------
    raw_present = load_tradingview_signal(
        settings.data_dir, settings.signal_path) is not None
    signal = load_tradingview_signal(
        settings.data_dir,
        settings.signal_path,
        max_age_hours=settings.max_age_hours,
    )
    if signal is None:
        if raw_present:
            raise NoSignalError(
                f"Signal is stale (> {settings.max_age_hours:.0f}h old or "
                "undatable) — treated as no signal. No verdict."
            )
        raise NoSignalError(
            f"No TradingView signal found in {settings.data_dir}. No verdict."
        )
    age_h = signal_age_hours(signal)

    ticker = (_as_str(signal.get("ticker") or signal.get("symbol"))
              or "UNKNOWN").upper()
    trend = (_as_str(signal.get("trend") or signal.get("direction"))
             or "").lower()
    rsi = _as_float(signal.get("momentum") or signal.get("rsi"))
    signal_price = _as_float(signal.get("price") or signal.get("close"))
    side = trend_to_side(trend) or "long"

    # ---- 3. calibration lookup ------------------------------------------
    drift = signal_to_drift(
        signal,
        ticker,
        horizon_days=settings.horizon_days,
        calibration_dir=settings.calibration_dir,
    )

    # ---- 4. market state (hard-fail on fallback) -------------------------
    market = _resolve_market(settings, ticker, signal_price)
    s0, sigma = market["s0"], market["sigma"]
    wsig = weekly_sigma(sigma)

    # ---- 5. decision layer ----------------------------------------------
    stop_pct = min(0.99, settings.stop_mult * wsig)
    tp_pct = (settings.tp_mult * wsig) if settings.tp_mult is not None else None
    rule = make_rule(stop_pct, tp_pct, side=side)

    be_paths = settings.be_paths or max(settings.paths // 2, 10_000)
    # Zero-drift run: what pure noise does to this structure.
    noise = run_structure(ticker, s0, sigma, 0.0, rule, settings.paths,
                          settings.cost, settings.seed)
    noise_stop_rate = float(noise.get("stop_hit_rate", float("nan")))
    noise_drag = float(np.mean(noise["pnl_pct"]))

    # Sweep stop multiples at zero drift so the chosen stop can be judged
    # against alternatives (P(stopped out by pure noise) per structure).
    noise_sweep_rows: List[Dict[str, float]] = []
    if settings.noise_sweep:
        for sm in settings.sweep_stop_mults:
            sp = min(0.99, float(sm) * wsig)
            if abs(sp - stop_pct) < 1e-12 and tp_pct is None:
                z = noise  # chosen structure already simulated
            else:
                z = run_structure(ticker, s0, sigma, 0.0,
                                  make_rule(sp, None, side=side),
                                  settings.paths, settings.cost,
                                  settings.seed)
            noise_sweep_rows.append({
                "stop_mult_weekly_sigma": float(sm),
                "stop_pct": sp,
                "noise_stop_rate": float(z.get("stop_hit_rate",
                                               float("nan"))),
                "noise_drag_pct": float(np.mean(z["pnl_pct"])),
            })

    be_annual = breakeven_edge(ticker, s0, sigma, rule, side, be_paths,
                               settings.cost, settings.seed)
    be_weekly = be_annual * settings.horizon_days / TRADING_DAYS_PER_YEAR

    # Simulate under the calibrated drift (used as-is; the rule engine
    # already handles short P&L against the underlying drift).
    st = run_structure(ticker, s0, sigma, drift.mu_annual, rule,
                       settings.paths, settings.cost, settings.seed)
    pnl = st["pnl_pct"]
    expectancy = float(np.mean(pnl))
    p_win = float(np.mean(pnl > 0))
    p5, p95 = (float(x) for x in np.percentile(pnl, [5, 95]))

    # Edge in the trade direction: a short trade profits from negative mu.
    edge_weekly = drift.mu_weekly if side == "long" else -drift.mu_weekly
    edge_annual = drift.mu_annual if side == "long" else -drift.mu_annual

    # ---- second witness: 12-1 momentum lens ------------------------------
    if settings.agreement_filter:
        mom_lens = momentum_lens_check(
            ticker, market["daily_log_returns"], side,
            settings.horizon_days, settings.calibration_dir)
    else:
        mom_lens = {"active": False, "agrees": None,
                    "note": "agreement filter disabled by settings"}

    # ---- kill-switch -----------------------------------------------------
    ks_tripped, ks_reason = outcome_tracker.check_kill_switch(
        settings.trade_log)

    # ---- verdict ---------------------------------------------------------
    verdict = "NO_TRADE"
    reason = ""
    stress_mult = float(settings.cost_stress_mult or 0.0)
    be_annual_stress: Optional[float] = None
    be_weekly_stress: Optional[float] = None
    if drift.source != "calibration":
        reason = f"no calibrated drift ({drift.reason})"
    elif edge_weekly <= 0.0:
        reason = (
            f"calibrated edge in trade direction is {fmt_pct(edge_weekly)} "
            f"weekly (bucket {drift.bucket}, t={drift.t_stat:.2f}, "
            f"n_eff={drift.n_eff:.0f}) — zero or adverse"
        )
    elif edge_weekly <= be_weekly:
        reason = (
            f"calibrated edge {fmt_pct(edge_weekly)} weekly ≤ breakeven "
            f"{fmt_pct(be_weekly)} for this structure"
        )
    elif expectancy <= 0.0:
        reason = (
            f"expectancy under calibrated drift is {fmt_pct(expectancy)} ≤ 0"
        )
    else:
        # Cost stress: the edge must also clear the breakeven computed at
        # stressed costs, so a slightly-worse-than-assumed fill can't turn
        # a marginal TRADE negative. (Computed lazily — only for edges that
        # already beat the base breakeven.)
        if stress_mult > 1.0:
            be_annual_stress = breakeven_edge(
                ticker, s0, sigma, rule, side, be_paths,
                settings.cost * stress_mult, settings.seed)
            be_weekly_stress = (be_annual_stress * settings.horizon_days
                                / TRADING_DAYS_PER_YEAR)
        if (be_weekly_stress is not None
                and edge_weekly <= be_weekly_stress):
            reason = (
                f"fails cost stress: edge {fmt_pct(edge_weekly)} weekly ≤ "
                f"breakeven {fmt_pct(be_weekly_stress)} at "
                f"{stress_mult:.1f}× costs "
                f"({settings.cost * stress_mult * 1e4:.0f} bps/side)"
            )
        else:
            verdict = "TRADE"
            reason = (
                f"edge {fmt_pct(edge_weekly)} weekly > breakeven "
                f"{fmt_pct(be_weekly)} (margin "
                f"{fmt_pct(edge_weekly - be_weekly)})"
                + (f"; survives {stress_mult:.1f}× cost stress"
                   if be_weekly_stress is not None else "")
            )

    if (verdict == "TRADE" and mom_lens.get("active")
            and not mom_lens.get("agrees")):
        verdict = "NO_TRADE"
        reason = f"agreement filter: {mom_lens['note']}"

    if verdict == "TRADE" and ks_tripped and not settings.override_killswitch:
        verdict = "NO_TRADE"
        reason = ks_reason

    # ---- sizing ----------------------------------------------------------
    f_kelly = kelly_fraction(pnl) if verdict == "TRADE" else 0.0
    notional_kelly = f_kelly * settings.account
    notional_riskcap = (
        (MAX_RISK_PER_TRADE * settings.account) / stop_pct
        if stop_pct > 0 else 0.0
    )
    notional = min(notional_kelly, notional_riskcap)
    shares = int(notional // s0) if s0 > 0 and verdict == "TRADE" else 0
    if verdict == "TRADE" and shares <= 0:
        verdict = "NO_TRADE"
        reason = "position size rounded to zero shares"
        f_kelly = 0.0
        notional = 0.0

    record: Dict[str, Any] = {
        "timestamp_utc": now.isoformat(),
        # Quarantined legacy engine: the bot's bridge only plans verdicts
        # from the canonical meta_label_v2 engine; this stamp makes the
        # drift pipeline's output explicit so it can never be confused
        # for (or executed as) a v2 decision.
        "engine": "legacy_drift_v1",
        "ticker": ticker,
        "verdict": verdict,
        "reason": reason,
        "side": side,
        "horizon_days": settings.horizon_days,
        # inputs
        "signal": {k: v for k, v in signal.items() if k != "raw"},
        "signal_received_at_utc": signal.get("received_at_utc"),
        "signal_age_hours": age_h,
        "trend": trend,
        "rsi": rsi,
        "drift_estimate": drift.as_dict(),
        "drift_label": drift.label(),
        "s0": s0,
        "annual_sigma": sigma,
        "weekly_sigma": wsig,
        "sigma_method": market["sigma_method"],
        "market_source": market["source"],
        "cost_per_side": settings.cost,
        "paths": settings.paths,
        "seed": settings.seed,
        # structure
        "structure": {
            "stop_mult_weekly_sigma": settings.stop_mult,
            "stop_pct": stop_pct,
            "tp_mult_weekly_sigma": settings.tp_mult,
            "tp_pct": tp_pct,
            "max_holding_days": settings.horizon_days,
        },
        "noise_stop_rate": noise_stop_rate,
        "noise_drag_pct": noise_drag,
        "noise_stop_sweep": noise_sweep_rows,
        # decision numbers
        "breakeven_mu_annual": be_annual,
        "breakeven_mu_weekly": be_weekly,
        "cost_stress_mult": stress_mult,
        "breakeven_mu_annual_stress": be_annual_stress,
        "breakeven_mu_weekly_stress": be_weekly_stress,
        "momentum_lens": mom_lens,
        "edge_weekly": edge_weekly,
        "edge_annual": edge_annual,
        "expectancy_pct": expectancy,
        "p_win": p_win,
        "pnl_p5": p5,
        "pnl_p95": p95,
        "stop_hit_rate": float(st.get("stop_hit_rate", float("nan"))),
        # sizing
        "sizing": {
            "kelly_cap": KELLY_CAP,
            "kelly_fraction": f_kelly,
            "notional_kelly": notional_kelly,
            "notional_risk_cap": notional_riskcap,
            "max_risk_per_trade": MAX_RISK_PER_TRADE,
            "notional": notional if verdict == "TRADE" else 0.0,
            "shares": shares,
            "account": settings.account,
        },
        "kill_switch": {"tripped": ks_tripped, "reason": ks_reason,
                        "overridden": settings.override_killswitch},
        "settled": False,
    }

    if settings.write_files:
        settings.verdict_dir.mkdir(parents=True, exist_ok=True)
        ts_slug = now.strftime("%Y%m%dT%H%M%SZ")
        verdict_path = settings.verdict_dir / f"{ts_slug}_{ticker}.json"
        verdict_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        record["verdict_path"] = str(verdict_path)
        outcome_tracker.log_verdict(record, settings.trade_log)

    return record


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def verdict_text(v: Dict[str, Any]) -> str:
    de = v["drift_estimate"]
    age = v.get("signal_age_hours")
    age_txt = f" age={age:.1f}h" if age is not None else ""
    if de["t_stat"] is not None and de["n_eff"] is not None:
        drift_tag = (f"[{de['bucket'] or 'n/a'}, t={de['t_stat']:.2f}, "
                     f"n_eff={de['n_eff']:.0f}]")
    else:
        drift_tag = f"[{de['bucket'] or 'n/a'}, source={de['source']}]"
    lines = [
        "=" * 64,
        f"WEEKLY VERDICT: {v['ticker']}  ({v['side']})  "
        f"[{v['timestamp_utc']}]",
        "=" * 64,
        f"Signal: trend={v['trend']} RSI={v['rsi']}{age_txt}",
        f"Market: S0={v['s0']:.2f}  σ={fmt_pct(v['annual_sigma'], 1)}/yr "
        f"({v['sigma_method']}; source={v['market_source']})",
        f"Drift:  {v['drift_label']}",
        f"Structure: stop {fmt_pct(v['structure']['stop_pct'])} "
        f"({v['structure']['stop_mult_weekly_sigma']:.1f}σw)"
        + (f", TP {fmt_pct(v['structure']['tp_pct'])}"
           if v['structure']['tp_pct'] else ", no TP")
        + f", max hold {v['horizon_days']}d, "
          f"cost {v['cost_per_side'] * 1e4:.0f} bps/side",
        f"Noise (zero drift): stop-out {fmt_pct(v['noise_stop_rate'], 1)}, "
        f"drag {fmt_pct(v['noise_drag_pct'])}/trade",
        *(
            [
                "Noise stop-out sweep (zero drift): "
                + "  ".join(
                    f"{r['stop_mult_weekly_sigma']:.1f}σw→"
                    f"{fmt_pct(r['noise_stop_rate'], 0)}"
                    for r in v["noise_stop_sweep"]
                )
            ]
            if v.get("noise_stop_sweep") else []
        ),
        f"Breakeven: μ={fmt_pct(v['breakeven_mu_annual'], 1)}/yr "
        f"≈ {fmt_pct(v['breakeven_mu_weekly'])}/week"
        + (
            f"  |  at {v['cost_stress_mult']:.1f}× costs: "
            f"{fmt_pct(v['breakeven_mu_weekly_stress'])}/week"
            if v.get("breakeven_mu_weekly_stress") is not None else ""
        ),
        *(
            [f"Momentum lens: {v['momentum_lens']['note']}"]
            if v.get("momentum_lens", {}).get("note") else []
        ),
        f"Under calibrated drift {drift_tag}: "
        f"expectancy {fmt_pct(v['expectancy_pct'])}/trade, "
        f"P(win) {fmt_pct(v['p_win'], 1)}, "
        f"5–95% [{fmt_pct(v['pnl_p5'])}, {fmt_pct(v['pnl_p95'])}]",
    ]
    ks = v["kill_switch"]
    if ks["tripped"]:
        lines.append(f"Kill-switch: TRIPPED"
                     + (" (OVERRIDDEN)" if ks["overridden"] else "")
                     + f" — {ks['reason']}")
    lines.append("-" * 64)
    lines.append(f"VERDICT: {v['verdict']} — {v['reason']}")
    if v["verdict"] == "TRADE":
        s = v["sizing"]
        lines.append(
            f"Size: {KELLY_CAP:.0%}-Kelly {fmt_pct(s['kelly_fraction'], 1)} "
            f"→ ${s['notional_kelly']:,.0f}; risk cap "
            f"({MAX_RISK_PER_TRADE:.0%} at stop) ${s['notional_risk_cap']:,.0f} "
            f"⇒ {s['shares']:,} shares (~${s['shares'] * v['s0']:,.0f})"
        )
    if v.get("verdict_path"):
        lines.append(f"Verdict JSON: {v['verdict_path']}")
    lines.append("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Production weekly decision from the latest TradingView "
                    "signal (calibrated drift, EWMA vol, breakeven, sizing).",
    )
    p.add_argument("--data-dir", default=str(DEFAULT_TV_DATA_DIR))
    p.add_argument("--signal-path", default=None)
    p.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    p.add_argument("--max-age-hours", type=float,
                   default=DEFAULT_MAX_SIGNAL_AGE_HOURS)
    p.add_argument("--paths", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost", type=float, default=DEFAULT_COST,
                   help="proportional cost per side (default 5 bps)")
    p.add_argument("--stop-mult", type=float, default=1.5)
    p.add_argument("--tp-mult", type=float, default=None)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--s0", type=float, default=None,
                   help="manual starting price (offline use)")
    p.add_argument("--sigma", type=float, default=None,
                   help="manual annualized vol (offline use)")
    p.add_argument("--trade-log",
                   default=str(outcome_tracker.DEFAULT_TRADE_LOG))
    p.add_argument("--verdict-dir", default=str(DEFAULT_VERDICT_DIR))
    p.add_argument("--override-killswitch", action="store_true")
    p.add_argument("--cost-stress-mult", type=float, default=2.0,
                   help="TRADE must also beat breakeven at this multiple of "
                        "--cost (default 2.0; <=1 disables)")
    p.add_argument("--no-agreement-filter", action="store_true",
                   help="disable the 12-1 momentum second-witness filter")
    # Demo helpers
    p.add_argument("--demo", action="store_true",
                   help="write a demo signal into --data-dir first (offline)")
    p.add_argument("--ticker", default="AAPL", help="demo ticker")
    p.add_argument("--price", type=float, default=190.0, help="demo price")
    p.add_argument("--trend", default="bullish",
                   choices=["bullish", "bearish"], help="demo trend")
    p.add_argument("--momentum", type=float, default=62.0, help="demo RSI")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.demo:
        path = write_demo_signal(
            data_dir / "latest_signal.json",
            ticker=args.ticker,
            price=args.price,
            trend=args.trend,
            momentum=args.momentum,
        )
        print(f"[demo] Wrote sample TradingView signal → {path}\n")

    settings = PipelineSettings(
        data_dir=data_dir,
        signal_path=Path(args.signal_path) if args.signal_path else None,
        calibration_dir=Path(args.calibration_dir),
        max_age_hours=float(args.max_age_hours),
        paths=int(args.paths),
        seed=int(args.seed),
        cost=float(args.cost),
        stop_mult=float(args.stop_mult),
        tp_mult=float(args.tp_mult) if args.tp_mult is not None else None,
        account=float(args.account),
        s0=args.s0,
        sigma=args.sigma,
        trade_log=Path(args.trade_log),
        verdict_dir=Path(args.verdict_dir),
        override_killswitch=bool(args.override_killswitch),
        cost_stress_mult=float(args.cost_stress_mult),
        agreement_filter=not args.no_agreement_filter,
    )

    try:
        verdict = run_pipeline(settings)
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    print(verdict_text(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
