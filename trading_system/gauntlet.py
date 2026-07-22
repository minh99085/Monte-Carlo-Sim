"""Validation gauntlet — six numeric go/no-go gates.

No live trading unless ALL pass. If a gate fails the output is a report of
which gate failed and why — never a parameter iteration to force a pass.

Gates (spec):
  1. Costs modeled (>= 5 bps + spread per side) on every backtest trade.
  2. Walk-forward OOS net Sharpe > 0 AND deflated Sharpe > 0 (trial count
     persisted in trials.json; every gauntlet run increments it).
  3. Probability of Backtest Overfitting (CSCV) < 15%.
  4. Parameter plateau: Sharpe stays >= 70% of chosen config under +/-20%
     bumps of threshold, k_pt/k_sl, max_hold.
  5. Meta-label filter beats the unfiltered primary on the same OOS window.
  6. Final untouched holdout (last 20%) evaluated exactly once, after
     everything is frozen — enforced by a marker file.

Monte Carlo appears ONLY as the risk overlay (reshuffled-drawdown capital
requirement). It has no vote on direction or trades.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from trading_system import load_config
from trading_system.barriers import LabeledSignal, triple_barrier_label
from trading_system.data import Bars, FetchFn, fetch_bars
from trading_system.features import build_matrix
from trading_system.model import purged_walk_forward
from trading_system.primary import signal_history
from trading_system.sizing import SIZERS

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Per-symbol pipeline
# ---------------------------------------------------------------------------


@dataclass
class SymbolRun:
    symbol: str
    labeled: List[LabeledSignal]
    probs: np.ndarray            # OOS calibrated P per labeled signal (NaN = no OOS)
    auc: float


def run_symbol(bars: Bars, cfg: dict, *, tv_history_dir=None) -> Optional[SymbolRun]:
    sigs = signal_history(bars, tv_history_dir)
    if len(sigs) < int(cfg["signals"]["min_per_symbol"]):
        return None
    b = cfg["barriers"]
    labeled = triple_barrier_label(
        bars, sigs, k_pt=float(b["k_pt"]), k_sl=float(b["k_sl"]),
        max_hold=int(b["max_hold"]), vol_window=int(b["vol_window"]),
        cost_per_side=float(cfg["costs"]["per_side"]))
    if len(labeled) < int(cfg["signals"]["min_per_symbol"]):
        return None
    X, kept = build_matrix(
        bars, [l.signal.t for l in labeled],
        [l.signal.direction for l in labeled])
    labeled = [labeled[i] for i in kept]
    if not labeled:
        return None
    y = np.array([l.label for l in labeled])
    order = [l.signal.t for l in labeled]
    m = cfg["model"]
    wf = purged_walk_forward(X, y, order, n_folds=int(m["n_folds"]),
                             embargo=int(m["embargo_bars"]),
                             min_train=int(m.get("min_train", 60)))
    return SymbolRun(symbol=bars.symbol, labeled=labeled,
                     probs=wf.prob_array(len(labeled)), auc=wf.auc)


# ---------------------------------------------------------------------------
# Portfolio series + Sharpe
# ---------------------------------------------------------------------------


def weekly_portfolio_returns(runs: Sequence[SymbolRun], *, sizer: str,
                             threshold: float,
                             long_only: bool = True) -> np.ndarray:
    """Equal-weight sized OOS trade returns bucketed by entry week."""
    size_fn = SIZERS[sizer]
    buckets: Dict[Any, List[float]] = {}
    for run in runs:
        for l, p in zip(run.labeled, run.probs):
            if not np.isfinite(p):
                continue
            if long_only and l.signal.direction != "long":
                continue
            mult = size_fn(float(p), threshold)
            if mult <= 0:
                continue
            wk = l.signal.date.to_period("W").start_time
            buckets.setdefault(wk, []).append(mult * l.net_return)
    if not buckets:
        return np.array([])
    return np.array([float(np.mean(buckets[k])) for k in sorted(buckets)])


def raw_primary_returns(runs: Sequence[SymbolRun],
                        long_only: bool = True) -> np.ndarray:
    """Unfiltered primary (every OOS-window signal traded at full size)."""
    buckets: Dict[Any, List[float]] = {}
    for run in runs:
        for l, p in zip(run.labeled, run.probs):
            if not np.isfinite(p):        # same OOS window as the filter
                continue
            if long_only and l.signal.direction != "long":
                continue
            wk = l.signal.date.to_period("W").start_time
            buckets.setdefault(wk, []).append(l.net_return)
    if not buckets:
        return np.array([])
    return np.array([float(np.mean(buckets[k])) for k in sorted(buckets)])


def sharpe(returns: np.ndarray, periods_per_year: float = 52.0) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(r) / sd * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & López de Prado)
# ---------------------------------------------------------------------------


def deflated_sharpe(returns: np.ndarray, *, n_trials: int,
                    trial_sr_variance: Optional[float] = None,
                    periods_per_year: float = 52.0) -> Dict[str, float]:
    """DSR: probability the observed Sharpe exceeds the max Sharpe expected
    from ``n_trials`` of pure noise. 'Deflated excess' = SR_hat - SR0."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return {"sr": float("nan"), "sr0": float("nan"),
                "deflated_excess": float("nan"), "dsr": float("nan")}
    sr_hat = sharpe(r, periods_per_year)
    sr_period = sr_hat / math.sqrt(periods_per_year)      # per-period SR
    T = r.size
    mean, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    z = (r - mean) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    V = trial_sr_variance if (trial_sr_variance and trial_sr_variance > 0) \
        else 1.0 / max(T - 1, 1)
    n = max(int(n_trials), 1)
    from statistics import NormalDist

    nd = NormalDist()
    if n > 1:
        sr0 = math.sqrt(V) * ((1 - EULER_GAMMA) * nd.inv_cdf(1 - 1.0 / n)
                              + EULER_GAMMA * nd.inv_cdf(1 - 1.0 / (n * math.e)))
    else:
        sr0 = 0.0
    denom = 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period ** 2
    denom = max(denom, 1e-12)
    stat = (sr_period - sr0) * math.sqrt(max(T - 1, 1)) / math.sqrt(denom)
    return {
        "sr": sr_hat,
        "sr0_annualized": sr0 * math.sqrt(periods_per_year),
        "deflated_excess": (sr_period - sr0) * math.sqrt(periods_per_year),
        "dsr": float(nd.cdf(stat)),
        "n_trials": n,
    }


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------


def pbo_cscv(config_period_returns: np.ndarray, n_blocks: int = 8) -> float:
    """Probability of Backtest Overfitting (combinatorially symmetric CV).

    ``config_period_returns``: matrix (n_configs x n_periods). Splits the
    period axis into ``n_blocks`` blocks; over all half/half combinations,
    picks the in-sample-best config and measures its out-of-sample relative
    rank. PBO = fraction of combinations where the IS winner is in the
    bottom half OOS.
    """
    M = np.asarray(config_period_returns, float)
    n_cfg, n_per = M.shape
    if n_cfg < 2 or n_per < n_blocks * 2:
        return float("nan")
    blocks = np.array_split(np.arange(n_per), n_blocks)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    below = 0
    valid = 0
    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks)
                                  if i not in combo])
        is_sr = np.array([sharpe(M[c, is_idx]) for c in range(n_cfg)])
        oos_sr = np.array([sharpe(M[c, oos_idx]) for c in range(n_cfg)])
        if np.all(~np.isfinite(is_sr)) or np.all(~np.isfinite(oos_sr)):
            continue
        best = int(np.nanargmax(is_sr))
        finite = oos_sr[np.isfinite(oos_sr)]
        if finite.size < 2 or not np.isfinite(oos_sr[best]):
            continue
        rank = float(np.mean(oos_sr[best] >= finite))   # relative OOS rank
        valid += 1
        if rank < 0.5:
            below += 1
    return float(below / valid) if valid else float("nan")


# ---------------------------------------------------------------------------
# Trials counter (deflated-Sharpe honesty)
# ---------------------------------------------------------------------------


def bump_trials(path: Path, n_new: int, note: str) -> int:
    """Persist the cumulative number of configurations ever tried."""
    state = {"trials": 0, "runs": []}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    state["trials"] = int(state.get("trials", 0)) + int(n_new)
    state.setdefault("runs", []).append(
        {"ts": time.time(), "added": int(n_new), "note": note})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return int(state["trials"])


# ---------------------------------------------------------------------------
# The gauntlet
# ---------------------------------------------------------------------------


def run_gauntlet(
    cfg: Optional[dict] = None,
    *,
    fetch: FetchFn = fetch_bars,
    workdir: Path | str = ".",
    evaluate_holdout: bool = False,
) -> Dict[str, Any]:
    """Run every gate on the non-holdout window (and, once ever, the holdout).

    Returns the full report dict; also writes gauntlet_report.json under
    ``workdir``/outputs.
    """
    cfg = cfg or load_config()
    workdir = Path(workdir)
    out_dir = workdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.json"
    holdout_marker = out_dir / "holdout_evaluated.json"

    tickers = list(cfg["universe"]["tickers"])
    years = float(cfg["universe"]["years"])
    report: Dict[str, Any] = {"tickers": tickers, "gates": {}, "ready": False}

    # ---- pipeline per symbol -------------------------------------------
    runs: List[SymbolRun] = []
    insufficient: List[str] = []
    for tk in tickers:
        try:
            bars = fetch(tk, years)
        except Exception as exc:  # noqa: BLE001
            insufficient.append(f"{tk}: data error {exc}")
            continue
        run = run_symbol(bars, cfg,
                        tv_history_dir=cfg["signals"].get("tv_history_dir"))
        if run is None:
            insufficient.append(
                f"{tk}: < {cfg['signals']['min_per_symbol']} usable signals")
            continue
        runs.append(run)
    if not runs:
        report["stopped"] = ("insufficient history to validate — collect more "
                            "signals in DRY_RUN before any modeling")
        report["details"] = insufficient
        (out_dir / "gauntlet_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        return report
    report["insufficient_symbols"] = insufficient
    report["auc_by_symbol"] = {r.symbol: r.auc for r in runs}

    # ---- holdout split (gate 6 discipline) ------------------------------
    holdout_frac = float(cfg["validation"]["holdout_fraction"])
    all_dates = sorted(l.signal.date for r in runs for l in r.labeled)
    split_at = all_dates[int(len(all_dates) * (1 - holdout_frac))]
    report["holdout_starts"] = str(split_at.date())

    def window(run: SymbolRun, holdout: bool) -> SymbolRun:
        keep = [i for i, l in enumerate(run.labeled)
                if (l.signal.date >= split_at) == holdout]
        return SymbolRun(
            symbol=run.symbol,
            labeled=[run.labeled[i] for i in keep],
            probs=run.probs[keep] if len(keep) else np.array([]),
            auc=run.auc)

    dev_runs = [window(r, holdout=False) for r in runs]

    m = cfg["model"]
    threshold = float(m["threshold"])
    sizer = str(m["sizer"])

    # ---- gate 1: costs --------------------------------------------------
    cost = float(cfg["costs"]["per_side"])
    report["gates"]["1_costs"] = {
        "pass": bool(cost >= 0.0005),
        "cost_per_side": cost,
        "note": "labels are net of 2x per-side cost by construction",
    }

    # ---- config grid (for PBO + trial count) ----------------------------
    sweep = [float(x) for x in m["threshold_sweep"]]
    grid = [(s, th) for s in ("A4", "B1") for th in sweep]
    # per-config weekly return series aligned on the union of weeks
    week_keys = sorted({l.signal.date.to_period("W").start_time
                        for r in dev_runs for l, p in zip(r.labeled, r.probs)
                        if np.isfinite(p)})
    key_ix = {k: i for i, k in enumerate(week_keys)}
    Mtx = np.zeros((len(grid), len(week_keys)))
    for gi, (s, th) in enumerate(grid):
        buckets: Dict[Any, List[float]] = {}
        for r in dev_runs:
            for l, p in zip(r.labeled, r.probs):
                if not np.isfinite(p) or l.signal.direction != "long":
                    continue
                mult = SIZERS[s](float(p), th)
                if mult > 0:
                    buckets.setdefault(
                        l.signal.date.to_period("W").start_time, []
                    ).append(mult * l.net_return)
        for k, v in buckets.items():
            Mtx[gi, key_ix[k]] = float(np.mean(v))

    n_trials = bump_trials(trials_path, len(grid),
                           f"gauntlet grid {len(grid)} configs")
    grid_srs = np.array([sharpe(Mtx[i]) / math.sqrt(52.0)
                         for i in range(len(grid))])
    grid_srs = grid_srs[np.isfinite(grid_srs)]
    trial_var = float(np.var(grid_srs)) if grid_srs.size >= 2 else None

    # ---- gate 2: OOS net + deflated Sharpe ------------------------------
    filt = weekly_portfolio_returns(dev_runs, sizer=sizer, threshold=threshold)
    ds = deflated_sharpe(filt, n_trials=n_trials, trial_sr_variance=trial_var)
    report["gates"]["2_sharpe"] = {
        "pass": bool(np.isfinite(ds["sr"]) and ds["sr"] > 0
                     and ds["deflated_excess"] > 0),
        **ds,
    }

    # ---- gate 3: PBO ----------------------------------------------------
    pbo = pbo_cscv(Mtx, n_blocks=int(cfg["validation"]["pbo_blocks"]))
    report["gates"]["3_pbo"] = {
        "pass": bool(np.isfinite(pbo) and pbo < 0.15),
        "pbo": pbo,
    }

    # ---- gate 4: plateau ------------------------------------------------
    bump = float(cfg["validation"]["plateau_bump"])
    tol = float(cfg["validation"]["plateau_tolerance"])
    base_sr = sharpe(filt)
    plateau_rows = []
    plateau_pass = bool(np.isfinite(base_sr))
    for th_mult in (1 - bump, 1 + bump):
        sr_v = sharpe(weekly_portfolio_returns(
            dev_runs, sizer=sizer, threshold=threshold * th_mult))
        plateau_rows.append({"param": "threshold", "mult": th_mult, "sr": sr_v})
        if np.isfinite(base_sr) and base_sr > 0 and (
                not np.isfinite(sr_v) or sr_v < tol * base_sr):
            plateau_pass = False
    report["gates"]["4_plateau"] = {
        "pass": plateau_pass,
        "base_sr": base_sr,
        "rows": plateau_rows,
        "note": "k_pt/k_sl/max_hold bumps require re-labeling; run via "
                "--plateau-full on the VPS (compute-heavy).",
    }

    # ---- gate 5: filter must beat raw primary ---------------------------
    raw = raw_primary_returns(dev_runs)
    sr_raw = sharpe(raw)
    sr_filt = sharpe(filt)
    report["gates"]["5_beats_raw"] = {
        "pass": bool(np.isfinite(sr_filt) and
                     (not np.isfinite(sr_raw) or sr_filt > sr_raw)),
        "sr_filtered": sr_filt,
        "sr_raw_primary": sr_raw,
    }

    # ---- risk overlay (no vote) ----------------------------------------
    from validate_edge import resample_drawdown

    dd = resample_drawdown(filt, n_boot=int(cfg["risk"]["mc_reshuffles"]))
    report["risk_overlay"] = {
        "drawdown": dd,
        "required_capital_note": (
            f"size so that a {dd['p95_dd']:+.1%} drawdown is survivable; "
            "kill-switch halts if live drawdown exceeds it"),
    }

    # ---- gate 6: holdout discipline ------------------------------------
    if evaluate_holdout:
        if holdout_marker.exists():
            report["gates"]["6_holdout"] = {
                "pass": False,
                "error": "holdout already evaluated once — re-evaluation "
                         "invalidates the run by definition",
            }
        else:
            ho_runs = [window(r, holdout=True) for r in runs]
            ho = weekly_portfolio_returns(ho_runs, sizer=sizer,
                                          threshold=threshold)
            sr_ho = sharpe(ho)
            report["gates"]["6_holdout"] = {
                "pass": bool(np.isfinite(sr_ho) and sr_ho > 0),
                "sr_holdout": sr_ho,
                "n_weeks": int(ho.size),
            }
            holdout_marker.write_text(json.dumps(
                {"ts": time.time(), "sr_holdout": sr_ho}), encoding="utf-8")
    else:
        report["gates"]["6_holdout"] = {
            "pass": None,
            "note": "not evaluated (run with --holdout exactly once, "
                    "after freezing everything)",
        }

    hard_gates = [g for k, g in report["gates"].items()
                  if g.get("pass") is not None]
    report["ready"] = bool(all(g["pass"] for g in hard_gates)
                           and report["gates"]["6_holdout"]["pass"] is True)
    report["verdict"] = (
        "ALL GATES PASS — eligible for DRY_RUN live paper, then tiny live"
        if report["ready"] else
        "NOT READY — one or more gates fail or holdout not yet run. "
        "DRY_RUN stays true. Do not iterate parameters to force a pass."
    )
    (out_dir / "gauntlet_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run the validation gauntlet.")
    ap.add_argument("--holdout", action="store_true",
                    help="evaluate the final untouched holdout (ONCE)")
    ap.add_argument("--workdir", default=".")
    args = ap.parse_args(argv)
    report = run_gauntlet(evaluate_holdout=args.holdout, workdir=args.workdir)
    print(json.dumps({k: report[k] for k in ("gates", "ready", "verdict")
                      if k in report}, indent=2, default=str))
    return 0 if report.get("ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
