"""Institutional-grade Investment Decision Report.

Turns the chunk-safe Monte Carlo engine in :mod:`mc_core` into a plain-English
risk report for a non-coder investor.  It runs the full institutional model
stack, a suite of stress tests, a model-risk confidence score, a lightweight
benchmark comparison and (best-effort) fundamentals, then renders a Markdown
report plus JSON/CSV exports.

Design rules:
* Reuses the existing chunk-safe :func:`mc_core.simulate` engine (never builds a
  full ``paths x steps`` matrix).
* No new heavy dependencies; ``yfinance`` is optional and used best-effort.
* Never emits guaranteed buy/sell advice -- only descriptive risk labels with
  explicit uncertainty caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional
import io
import json
import os

import numpy as np

import mc_core

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_TOLERANCES = ("Conservative", "Moderate", "Aggressive")

# Acceptable probability of a >20% loss, by risk tolerance.
_TOLERANCE_BIG_LOSS_CAP = {
    "Conservative": 0.08,
    "Moderate": 0.18,
    "Aggressive": 0.30,
}

# Descriptive investment labels (never a direct buy/sell command).
LABEL_FAVORABLE = "Favorable but risky"
LABEL_NEUTRAL = "Neutral / uncertain"
LABEL_UNFAVORABLE = "Unfavorable risk-reward"
LABEL_TOO_UNCERTAIN = "Too uncertain to judge"
INVESTMENT_LABELS = (
    LABEL_FAVORABLE, LABEL_NEUTRAL, LABEL_UNFAVORABLE, LABEL_TOO_UNCERTAIN,
)

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_LEVELS = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW)

# The full institutional model stack (all available models).
INSTITUTIONAL_MODELS = tuple(mc_core.MODELS)

DEFAULT_BENCHMARK = "SPY"

# Required plain-English report sections (order preserved in Markdown).
REPORT_SECTIONS = (
    "Bottom Line",
    "What the Simulation Says",
    "Best Case",
    "Base Case",
    "Bad Case",
    "Severe Stress Case",
    "Biggest Risks",
    "What Could Make This Wrong",
    "Position Sizing Warning",
    "Suggested Next Questions Before Investing",
)

DISCLAIMER = (
    "This is a statistical risk simulation, not investment advice, a forecast, "
    "or a guarantee. Past data and model assumptions can be wrong. Do your own "
    "research and consider consulting a licensed financial professional."
)


# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------


@dataclass
class InvestmentReportConfig:
    ticker: str = "AAPL"
    horizon: int = 252
    paths: int = 100_000
    chunk_size: int = mc_core.DEFAULT_SERIOUS_CHUNK_SIZE
    seed: Optional[int] = 42
    drift_mode: str = mc_core.DRIFT_HISTORICAL
    risk_tolerance: str = "Moderate"
    max_acceptable_loss_pct: float = 0.20     # e.g. 0.20 == 20% loss
    ruin_threshold: float = 0.50              # ruin if value < 50% of start
    investment_amount: float = 10_000.0
    benchmark: str = DEFAULT_BENCHMARK
    years: float = 5.0
    models: Optional[List[str]] = None        # default: full institutional stack

    def validate(self) -> "InvestmentReportConfig":
        if self.risk_tolerance not in RISK_TOLERANCES:
            raise ValueError(f"risk_tolerance must be one of {RISK_TOLERANCES}")
        if not (0.0 < self.max_acceptable_loss_pct < 1.0):
            raise ValueError("max_acceptable_loss_pct must be in (0, 1)")
        if not (0.0 < self.ruin_threshold < 1.0):
            raise ValueError("ruin_threshold must be in (0, 1)")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.paths < 1:
            raise ValueError("paths must be >= 1")
        return self


# ---------------------------------------------------------------------------
# Stress-test suite
# ---------------------------------------------------------------------------


def stress_scenarios() -> List[Dict[str, Any]]:
    """Institutional stress overlays as (name, config-override) pairs."""
    return [
        {"name": "Zero drift", "overrides": {"drift_mode": mc_core.DRIFT_ZERO}},
        {"name": "Half historical drift", "overrides": {"drift_mode": mc_core.DRIFT_HALF}},
        {"name": "Volatility doubled",
         "overrides": {"stress_enabled": True, "stress_vol_multiplier": 2.0}},
        {"name": "One-day crash -10%",
         "overrides": {"stress_enabled": True, "stress_crash_pct": 0.10}},
        {"name": "One-day crash -20%",
         "overrides": {"stress_enabled": True, "stress_crash_pct": 0.20}},
        {"name": "Bear regime",
         "overrides": {"model": mc_core.MODEL_REGIME, "regime_preset": "stock",
                       "drift_mode": mc_core.DRIFT_ZERO}},
        {"name": "Combined stress (zero drift + 2x vol + -20% crash)",
         "overrides": {"drift_mode": mc_core.DRIFT_ZERO, "stress_enabled": True,
                       "stress_vol_multiplier": 2.0, "stress_crash_pct": 0.20}},
    ]


def run_stress_suite(base_config: mc_core.SimulationConfig) -> Dict[str, Any]:
    """Run the stress overlays on the base config; return rows + worst scenario."""
    rows: List[Dict[str, Any]] = []
    for scenario in stress_scenarios():
        cfg = replace(base_config, **scenario["overrides"])
        result = mc_core.simulate(cfg)
        s = result.stats
        rows.append({
            "scenario": scenario["name"],
            "expected_value": s["expected_value"],
            "median_value": s["median_value"],
            "prob_loss": s["prob_loss"],
            "prob_loss_20": s["prob_loss_20"],
            "prob_ruin": s.get("prob_ruin", float("nan")),
            "chunk_safe": result.memory.is_chunk_safe,
        })
    # Worst scenario = lowest expected value (most damaging on average).
    worst = min(rows, key=lambda r: r["expected_value"]) if rows else None
    return {
        "rows": rows,
        "worst_scenario": worst["scenario"] if worst else None,
        "combined": next((r for r in rows if r["scenario"].startswith("Combined")), None),
    }


# ---------------------------------------------------------------------------
# Model-risk score / confidence
# ---------------------------------------------------------------------------


def model_risk_score(
    *,
    hist_length: int,
    effective_mu: float,
    sigma: float,
    model_rows: List[Dict[str, Any]],
    median_prob_loss_20: float,
    backtest_coverage: Optional[float] = None,
    evt_available: bool = True,
    stress_available: bool = True,
    any_model_failed: bool = False,
) -> Dict[str, Any]:
    """Compute a simple model-risk score -> High / Medium / Low confidence."""
    risk_points = 0
    warnings: List[str] = []

    if hist_length < mc_core.TRADING_DAYS_PER_YEAR:
        risk_points += 2
        warnings.append(
            f"Very short history ({hist_length} days); estimates are unreliable.")
    elif hist_length < 2 * mc_core.TRADING_DAYS_PER_YEAR:
        risk_points += 1
        warnings.append(f"Limited history ({hist_length} days).")

    if effective_mu > 0.50:
        risk_points += 2
        warnings.append(
            f"Estimated drift ({effective_mu:.0%}/yr) is extreme; likely over-optimistic.")
    elif effective_mu > 0.30:
        risk_points += 1
        warnings.append(f"Estimated drift ({effective_mu:.0%}/yr) is high.")

    if sigma > 0.60:
        risk_points += 2
        warnings.append(f"Very high volatility ({sigma:.0%}/yr).")
    elif sigma > 0.40:
        risk_points += 1
        warnings.append(f"Elevated volatility ({sigma:.0%}/yr).")

    # Model disagreement on the >20% loss probability.
    if model_rows:
        pl20 = [r["prob_loss_20"] for r in model_rows]
        spread = max(pl20) - min(pl20)
        if spread > 0.15:
            risk_points += 2
            warnings.append(
                f"Models disagree strongly on big-loss risk (spread {spread:.0%}).")
        elif spread > 0.08:
            risk_points += 1
            warnings.append(f"Models disagree on big-loss risk (spread {spread:.0%}).")

    # Tail severity.
    if median_prob_loss_20 > 0.25:
        risk_points += 2
        warnings.append(
            f"Severe tail risk: ~{median_prob_loss_20:.0%} chance of a >20% loss.")
    elif median_prob_loss_20 > 0.15:
        risk_points += 1
        warnings.append(
            f"Notable tail risk: ~{median_prob_loss_20:.0%} chance of a >20% loss.")

    if backtest_coverage is not None and not (0.80 <= backtest_coverage <= 0.98):
        risk_points += 1
        warnings.append(
            f"Backtest coverage off-target ({backtest_coverage:.0%}); model fit is questionable.")

    if not evt_available:
        risk_points += 1
        warnings.append("EVT tail analysis unavailable.")
    if not stress_available:
        risk_points += 1
        warnings.append("Stress tests unavailable.")
    if any_model_failed:
        risk_points += 3
        warnings.append("One or more models failed to run; treat results with caution.")

    if risk_points <= 2:
        confidence = CONFIDENCE_HIGH
    elif risk_points <= 5:
        confidence = CONFIDENCE_MEDIUM
    else:
        confidence = CONFIDENCE_LOW

    if confidence == CONFIDENCE_LOW:
        warnings.append("Do not rely on this result alone.")

    return {"confidence": confidence, "risk_points": int(risk_points),
            "warnings": warnings}


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


def _max_drawdown_from_returns(daily_log_returns: np.ndarray) -> float:
    r = np.asarray(daily_log_returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.exp(np.cumsum(r))
    running_max = np.maximum.accumulate(equity)
    dd = 1.0 - equity / running_max
    return float(np.max(dd)) if dd.size else 0.0


def _annual_stats(daily_log_returns: np.ndarray) -> Dict[str, float]:
    r = np.asarray(daily_log_returns, dtype=float)
    n = r.size
    daily_mu = float(np.mean(r)) if n else 0.0
    daily_sd = float(np.std(r, ddof=1)) if n > 1 else 0.0
    return {
        "annual_return": daily_mu * mc_core.TRADING_DAYS_PER_YEAR,
        "annual_volatility": daily_sd * np.sqrt(mc_core.TRADING_DAYS_PER_YEAR),
        "max_drawdown": _max_drawdown_from_returns(r),
    }


def benchmark_comparison(
    ticker_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    ticker: str = "TICKER",
    benchmark: str = DEFAULT_BENCHMARK,
) -> Dict[str, Any]:
    """Lightweight ticker-vs-benchmark comparison (beta, corr, excess, downside)."""
    a = np.asarray(ticker_returns, dtype=float).ravel()
    b = np.asarray(benchmark_returns, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return {"available": False, "note": "Insufficient return history for benchmark."}
    m = min(a.size, b.size)
    a, b = a[-m:], b[-m:]

    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(a, b, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")

    tick = _annual_stats(a)
    bench = _annual_stats(b)
    downside_a = a[a < 0]
    downside_b = b[b < 0]
    dd_a = float(np.sqrt(np.mean(downside_a ** 2))) if downside_a.size else 0.0
    dd_b = float(np.sqrt(np.mean(downside_b ** 2))) if downside_b.size else 0.0

    return {
        "available": True,
        "ticker": ticker,
        "benchmark": benchmark,
        "aligned_days": int(m),
        "ticker_annual_return": tick["annual_return"],
        "benchmark_annual_return": bench["annual_return"],
        "ticker_annual_volatility": tick["annual_volatility"],
        "benchmark_annual_volatility": bench["annual_volatility"],
        "ticker_max_drawdown": tick["max_drawdown"],
        "benchmark_max_drawdown": bench["max_drawdown"],
        "beta": beta,
        "correlation": corr,
        "excess_return": tick["annual_return"] - bench["annual_return"],
        "ticker_downside_dev": dd_a,
        "benchmark_downside_dev": dd_b,
        "downside_vs_benchmark": dd_a - dd_b,
    }


# ---------------------------------------------------------------------------
# Fundamentals (best-effort, never raises)
# ---------------------------------------------------------------------------


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """Best-effort fundamentals via yfinance; always returns a dict, never raises."""
    result: Dict[str, Any] = {"available": False, "fields": {}}
    try:  # pragma: no cover - network/environment dependent
        import yfinance as yf
        info = {}
        tk = yf.Ticker(ticker)
        getter = getattr(tk, "get_info", None)
        info = getter() if callable(getter) else getattr(tk, "info", {}) or {}
        wanted = {
            "market_cap": "marketCap",
            "trailing_pe": "trailingPE",
            "forward_pe": "forwardPE",
            "revenue_growth": "revenueGrowth",
            "profit_margin": "profitMargins",
        }
        fields = {}
        for out_key, src_key in wanted.items():
            val = info.get(src_key)
            if val is not None:
                fields[out_key] = val
        if fields:
            result["available"] = True
            result["fields"] = fields
    except Exception as exc:  # noqa: BLE001
        result["note"] = f"Fundamental data unavailable from source ({type(exc).__name__})."
    if not result["available"] and "note" not in result:
        result["note"] = "Fundamental data unavailable from source."
    return result


# ---------------------------------------------------------------------------
# Investment label
# ---------------------------------------------------------------------------


def investment_label(
    *,
    prob_profit: float,
    prob_exceed_max_loss: float,
    risk_tolerance: str,
    confidence: str,
) -> str:
    """Map probabilities + tolerance + confidence to a descriptive label."""
    if confidence == CONFIDENCE_LOW:
        return LABEL_TOO_UNCERTAIN
    cap = _TOLERANCE_BIG_LOSS_CAP.get(risk_tolerance, 0.18)
    if prob_profit >= 0.55 and prob_exceed_max_loss <= cap:
        return LABEL_FAVORABLE
    if prob_profit < 0.45 or prob_exceed_max_loss > cap * 1.75:
        return LABEL_UNFAVORABLE
    return LABEL_NEUTRAL


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------


def _median(values: List[float]) -> float:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def build_investment_report(
    config: InvestmentReportConfig,
    *,
    market: Optional[mc_core.MarketParameters] = None,
    benchmark_market: Optional[mc_core.MarketParameters] = None,
    include_fundamentals: bool = True,
) -> Dict[str, Any]:
    """Run the full institutional report and return a serialisable dict.

    ``market``/``benchmark_market`` can be injected (e.g. in tests) to avoid
    network access; otherwise they are fetched (with offline fallback).
    """
    cfg = config.validate()
    models = list(cfg.models) if cfg.models else list(INSTITUTIONAL_MODELS)

    if market is None:
        market = mc_core.estimate_parameters_from_history(cfg.ticker, years=cfg.years)

    base_config = mc_core.SimulationConfig(
        ticker=cfg.ticker, s0=market.s0, paths=cfg.paths, horizon=cfg.horizon,
        mu=market.mu, sigma=market.sigma, chunk_size=cfg.chunk_size, seed=cfg.seed,
        drift_mode=cfg.drift_mode, historical_returns=market.daily_log_returns,
        ruin_threshold=cfg.ruin_threshold, sample_paths=50,
    )

    s0 = base_config.s0
    max_loss_level = s0 * (1.0 - cfg.max_acceptable_loss_pct)

    # ---- Institutional model stack (single pass, chunk-safe) ----------
    rows: List[Dict[str, Any]] = []
    per_model: Dict[str, Any] = {}
    model_stats: Dict[str, Any] = {}
    prob_exceed_by_model: List[float] = []
    failed_models: List[str] = []

    for model in models:
        try:
            mcfg = replace(base_config, model=model)
            result = mc_core.simulate(mcfg)
            evt = mc_core.evt_from_result(result)
            row = mc_core.comparison_row(result, evt=evt)
            prob_exceed = float(np.mean(result.final_values < max_loss_level))
            row["prob_exceed_max_loss"] = prob_exceed
            prob_exceed_by_model.append(prob_exceed)
            rows.append(row)
            model_stats[model] = result.stats
            per_model[model] = {
                "assumptions": mc_core.model_assumptions(mcfg, market),
                "statistics": result.stats,
                "evt": evt,
                "memory": {"is_chunk_safe": result.memory.is_chunk_safe},
            }
        except Exception as exc:  # noqa: BLE001 - record failure, keep going
            failed_models.append(model)
            per_model[model] = {"error": f"{type(exc).__name__}: {exc}"}

    if not rows:
        raise RuntimeError("All models failed to run for the investment report.")

    mc_core._assign_disagreement_ranks(rows)
    comparison = mc_core.ComparisonReport(
        base_config=base_config, rows=rows, per_model=per_model,
        most_conservative=mc_core.most_conservative_model(rows),
        market_source=market.source,
    )

    # ---- Consensus (cross-model medians) ------------------------------
    central = {
        "prob_profit": _median([r["prob_profit"] for r in rows]),
        "prob_loss": _median([r["prob_loss"] for r in rows]),
        "prob_gain_20": _median([r["prob_gain_20"] for r in rows]),
        "prob_loss_10": _median([r["prob_loss_10"] for r in rows]),
        "prob_loss_20": _median([r["prob_loss_20"] for r in rows]),
        "prob_drawdown_50": _median([r["prob_drawdown_50"] for r in rows]),
        "prob_ruin": _median([r["prob_ruin"] for r in rows]),
        "prob_exceed_max_loss": _median(prob_exceed_by_model),
        "median_ratio": _median([model_stats[m]["median_value"] / s0 for m in model_stats]),
        "expected_ratio": _median([model_stats[m]["expected_value"] / s0 for m in model_stats]),
        "p5_ratio": _median([model_stats[m]["percentiles"]["5"] / s0 for m in model_stats]),
        "p95_ratio": _median([model_stats[m]["percentiles"]["95"] / s0 for m in model_stats]),
    }
    worst_model = comparison.most_conservative

    # ---- Stress suite -------------------------------------------------
    stress = run_stress_suite(base_config)
    combined = stress.get("combined") or {}
    severe_ratio = (combined.get("median_value", s0) / s0) if combined else float("nan")

    # ---- Backtest (validation) ---------------------------------------
    backtest = None
    if market.daily_log_returns is not None and np.asarray(market.daily_log_returns).size > 2:
        prices = s0 * np.exp(np.cumsum(np.asarray(market.daily_log_returns)))
        backtest = mc_core.backtest_percentile_bands(prices, horizon=cfg.horizon)

    # ---- Model-risk score / confidence -------------------------------
    hist_len = int(np.asarray(market.daily_log_returns).size) if market.daily_log_returns is not None else 0
    risk = model_risk_score(
        hist_length=hist_len,
        effective_mu=base_config.effective_mu(),
        sigma=base_config.effective_sigma(),
        model_rows=rows,
        median_prob_loss_20=central["prob_loss_20"],
        backtest_coverage=(backtest.get("coverage") if backtest else None),
        evt_available=True,
        stress_available=True,
        any_model_failed=bool(failed_models),
    )
    confidence = risk["confidence"]

    # ---- Benchmark ----------------------------------------------------
    if benchmark_market is None and cfg.benchmark:
        benchmark_market = mc_core.estimate_parameters_from_history(
            cfg.benchmark, years=cfg.years)
    benchmark = {"available": False}
    if (benchmark_market is not None and market.daily_log_returns is not None
            and benchmark_market.daily_log_returns is not None):
        benchmark = benchmark_comparison(
            market.daily_log_returns, benchmark_market.daily_log_returns,
            ticker=cfg.ticker, benchmark=cfg.benchmark,
        )

    # ---- Fundamentals -------------------------------------------------
    fundamentals = fetch_fundamentals(cfg.ticker) if include_fundamentals else {
        "available": False, "note": "Fundamental data not requested."}

    # ---- Label --------------------------------------------------------
    label = investment_label(
        prob_profit=central["prob_profit"],
        prob_exceed_max_loss=central["prob_exceed_max_loss"],
        risk_tolerance=cfg.risk_tolerance,
        confidence=confidence,
    )

    # ---- Amount projections ------------------------------------------
    amt = cfg.investment_amount
    projections = {
        "best_case_value": amt * central["p95_ratio"],
        "base_case_value": amt * central["median_ratio"],
        "bad_case_value": amt * central["p5_ratio"],
        "severe_case_value": amt * severe_ratio if np.isfinite(severe_ratio) else float("nan"),
        "investment_amount": amt,
    }

    report = {
        "schema": "investment_report_v1",
        "math_model_version": mc_core.MATH_MODEL_VERSION,
        "inputs": {
            "ticker": cfg.ticker, "horizon": cfg.horizon, "paths": cfg.paths,
            "chunk_size": cfg.chunk_size, "seed": cfg.seed,
            "drift_mode": cfg.drift_mode, "risk_tolerance": cfg.risk_tolerance,
            "max_acceptable_loss_pct": cfg.max_acceptable_loss_pct,
            "ruin_threshold": cfg.ruin_threshold,
            "investment_amount": cfg.investment_amount,
            "benchmark": cfg.benchmark, "years": cfg.years,
            "models": models,
        },
        "market": {"source": market.source, "s0": s0,
                   "mu": market.mu, "sigma": market.sigma,
                   "history_days": hist_len},
        "investment_label": label,
        "model_confidence": confidence,
        "model_risk": risk,
        "central": central,
        "worst_model": worst_model,
        "projections": projections,
        "comparison_rows": rows,
        "per_model": per_model,
        "failed_models": failed_models,
        "stress_tests": stress,
        "benchmark_comparison": benchmark,
        "fundamentals": fundamentals,
        "backtest": backtest,
        "all_chunk_safe": comparison.all_chunk_safe,
        "disclaimer": DISCLAIMER,
    }
    report["plain_english"] = plain_english_sections(report)
    report["_comparison"] = comparison  # internal handle for CSV export (not serialised)
    return report


# ---------------------------------------------------------------------------
# Plain-English narrative
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%" if np.isfinite(x) else "n/a"


def _money(x: float) -> str:
    return f"${x:,.0f}" if np.isfinite(x) else "n/a"


def plain_english_sections(report: Dict[str, Any]) -> Dict[str, str]:
    """Generate the required plain-English sections as a name -> text dict."""
    c = report["central"]
    inp = report["inputs"]
    proj = report["projections"]
    paths = inp["paths"]
    horizon = inp["horizon"]
    ticker = inp["ticker"]
    label = report["investment_label"]
    confidence = report["model_confidence"]
    worst = report["worst_model"]
    amt = inp["investment_amount"]
    maxloss = inp["max_acceptable_loss_pct"]

    horizon_years = horizon / mc_core.TRADING_DAYS_PER_YEAR
    horizon_text = (f"{horizon_years:.1f}-year" if horizon_years >= 1
                    else f"{horizon}-trading-day")

    sections: Dict[str, str] = {}

    sections["Bottom Line"] = (
        f"{ticker} looks **{label}** for a **{inp['risk_tolerance'].lower()}** investor over a "
        f"{horizon_text} horizon. Across {paths:,} simulated outcomes, it made money in about "
        f"{_pct(c['prob_profit'])} of cases, while about {_pct(c['prob_loss_20'])} of cases lost "
        f"more than 20%. Model confidence is **{confidence}**. "
        + ("Because confidence is Low, do not rely on this result alone. "
           if confidence == CONFIDENCE_LOW else "")
        + "This is not advice or a prediction."
    )

    sections["What the Simulation Says"] = (
        f"Out of {paths:,} simulated {horizon_text} outcomes, {ticker} ended higher in about "
        f"{_pct(c['prob_profit'])} of them and lower in about {_pct(c['prob_loss'])}. "
        f"About {_pct(c['prob_gain_20'])} gained more than 20%, about {_pct(c['prob_loss_10'])} "
        f"lost more than 10%, and about {_pct(c['prob_loss_20'])} lost more than 20%. "
        f"Roughly {_pct(c['prob_exceed_max_loss'])} of outcomes breached your stated maximum "
        f"acceptable loss of {_pct(maxloss)}."
    )

    sections["Best Case"] = (
        f"In a good outcome (around the 95th percentile), a {_money(amt)} investment grows to about "
        f"{_money(proj['best_case_value'])}. Good cases are possible but not the most likely result."
    )
    sections["Base Case"] = (
        f"In a typical (median) outcome, {_money(amt)} ends near {_money(proj['base_case_value'])}. "
        "Half of outcomes are better than this and half are worse."
    )
    sections["Bad Case"] = (
        f"In a poor outcome (around the 5th percentile), {_money(amt)} falls to about "
        f"{_money(proj['bad_case_value'])}. Outcomes this bad or worse happened in roughly 1 in 20 "
        "simulations."
    )
    sections["Severe Stress Case"] = (
        "Under a combined stress test (no growth, doubled volatility, and a one-day 20% crash), a "
        f"typical outcome for {_money(amt)} is about {_money(proj['severe_case_value'])}. "
        f"The most damaging stress scenario overall was "
        f"\"{report['stress_tests'].get('worst_scenario', 'n/a')}\"."
    )

    big_risk_bits = [
        f"about {_pct(c['prob_loss_20'])} chance of losing more than 20%",
        f"about {_pct(c['prob_drawdown_50'])} chance of a 50% drawdown at some point",
        f"about {_pct(c['prob_ruin'])} chance of falling below your ruin threshold",
    ]
    sections["Biggest Risks"] = (
        "The main risks are: " + "; ".join(big_risk_bits) + ". "
        f"The most pessimistic model in the stack was {worst}, which you should weigh heavily if you "
        "are risk-averse."
    )

    warn_text = " ".join(report["model_risk"]["warnings"]) or "No major model-risk flags were raised."
    sections["What Could Make This Wrong"] = (
        "Simulations assume the future resembles the past and that the chosen models are correct. "
        "Real markets can behave very differently (regime changes, crashes, company-specific news). "
        f"Model-risk notes: {warn_text}"
    )

    sections["Position Sizing Warning"] = (
        "Do not size a position from a single number. Even a favorable label can lose money. "
        "A common guideline is to risk only what you can afford to lose and to diversify. "
        "The simulated Kelly fraction (if shown) is theoretical only and should not be used directly "
        "for sizing."
    )

    sections["Suggested Next Questions Before Investing"] = (
        "- Does this fit your time horizon and how much loss you can truly tolerate?\n"
        "- How does it compare to a simple benchmark like SPY on return and drawdown?\n"
        "- Are the company fundamentals and valuation reasonable?\n"
        "- What would you do if it dropped 20% next week?\n"
        "- Are you diversified, or is this too large a share of your portfolio?"
    )
    return sections


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the full plain-English Markdown report for a non-coder."""
    inp = report["inputs"]
    c = report["central"]
    buf = io.StringIO()
    w = buf.write

    w(f"# Investment Risk Report: {inp['ticker']}\n\n")
    w(f"**Investment label:** {report['investment_label']}  \n")
    w(f"**Model confidence:** {report['model_confidence']}  \n")
    w(f"**Horizon:** {inp['horizon']} trading days  |  **Simulated paths:** {inp['paths']:,}  |  "
      f"**Risk tolerance:** {inp['risk_tolerance']}  \n")
    w(f"**Data source:** {report['market']['source']}  |  "
      f"**Worst model:** {report['worst_model']}\n\n")

    if report["model_confidence"] == CONFIDENCE_LOW:
        w("> **Warning:** Model confidence is Low. Do not rely on this result alone.\n\n")

    # Quick risk cards.
    w("## Risk Snapshot\n\n")
    w("| Card | Value |\n|---|---|\n")
    w(f"| Profit Chance | {_pct(c['prob_profit'])} |\n")
    w(f"| Big Loss Chance (>20%) | {_pct(c['prob_loss_20'])} |\n")
    w(f"| Severe Drawdown Chance (50%) | {_pct(c['prob_drawdown_50'])} |\n")
    w(f"| Worst Model | {report['worst_model']} |\n")
    w(f"| Model Confidence | {report['model_confidence']} |\n")
    w(f"| Investment Label | {report['investment_label']} |\n\n")

    # Narrative sections in the required order.
    for name in REPORT_SECTIONS:
        text = report["plain_english"].get(name, "")
        w(f"## {name}\n\n{text}\n\n")

    # Stress tests.
    w("## Stress Test Details\n\n")
    w("| Scenario | Expected | Median | P(loss) | P(loss>20%) | P(ruin) |\n")
    w("|---|---|---|---|---|---|\n")
    for r in report["stress_tests"]["rows"]:
        w(f"| {r['scenario']} | {r['expected_value']:.2f} | {r['median_value']:.2f} | "
          f"{_pct(r['prob_loss'])} | {_pct(r['prob_loss_20'])} | {_pct(r['prob_ruin'])} |\n")
    w("\n")

    # Benchmark.
    bm = report["benchmark_comparison"]
    w("## Benchmark Comparison\n\n")
    if bm.get("available"):
        w(f"- Benchmark: **{bm['benchmark']}** (aligned {bm['aligned_days']} days)\n")
        w(f"- {inp['ticker']} annual return {_pct(bm['ticker_annual_return'])} vs "
          f"{bm['benchmark']} {_pct(bm['benchmark_annual_return'])} "
          f"(excess {_pct(bm['excess_return'])})\n")
        w(f"- Volatility {_pct(bm['ticker_annual_volatility'])} vs "
          f"{_pct(bm['benchmark_annual_volatility'])}\n")
        w(f"- Max drawdown {_pct(bm['ticker_max_drawdown'])} vs "
          f"{_pct(bm['benchmark_max_drawdown'])}\n")
        w(f"- Beta {bm['beta']:.2f}  |  Correlation {bm['correlation']:.2f}\n\n")
    else:
        w("Benchmark comparison unavailable.\n\n")

    # Fundamentals.
    f = report["fundamentals"]
    w("## Fundamentals (sanity check)\n\n")
    if f.get("available"):
        for k, v in f["fields"].items():
            w(f"- {k.replace('_', ' ').title()}: {v}\n")
        w("\n")
    else:
        w(f"{f.get('note', 'Fundamental data unavailable from source.')}\n\n")

    w(f"---\n\n_{report['disclaimer']}_\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def report_to_json(report: Dict[str, Any], *, indent: int = 2) -> str:
    """Serialise the institutional report to JSON (excludes internal handles)."""
    serialisable = {k: v for k, v in report.items() if not k.startswith("_")}
    return json.dumps(serialisable, indent=indent, default=mc_core._json_default)


def comparison_csv(report: Dict[str, Any]) -> str:
    """Institutional model-comparison CSV (reuses the core comparison writer)."""
    comp = report.get("_comparison")
    if comp is not None:
        return mc_core.comparison_to_csv(comp)
    # Fallback: build CSV directly from rows.
    buf = io.StringIO()
    if report["comparison_rows"]:
        cols = list(report["comparison_rows"][0].keys())
        buf.write(",".join(cols) + "\n")
        for r in report["comparison_rows"]:
            buf.write(",".join(str(r[c]) for c in cols) + "\n")
    return buf.getvalue()


def write_investment_report(report: Dict[str, Any], outdir: str = "outputs") -> Dict[str, str]:
    """Write Markdown, JSON and CSV exports to ``outdir``; returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    ticker = str(report["inputs"]["ticker"]).replace(os.sep, "_") or "ASSET"
    paths: Dict[str, str] = {}

    md_path = os.path.join(outdir, f"{ticker}_investment_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    paths["markdown"] = md_path

    json_path = os.path.join(outdir, f"{ticker}_investment_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(report_to_json(report))
    paths["json"] = json_path

    csv_path = os.path.join(outdir, f"{ticker}_institutional_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(comparison_csv(report))
    paths["csv"] = csv_path

    return paths
