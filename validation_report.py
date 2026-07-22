#!/usr/bin/env python3
"""
validation_report.py — render a run_full() result dict into VALIDATION.md.

The verdict paragraph is computed from the numbers, not written by hand: the
edge "survives" only if, after executable fills + mid-level (0.20%/side) cost
+ tax, the strategy's after-tax risk-adjusted return beats the benchmark's.
If it does not, the report says so at the top, unsoftened.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict


def _pct(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.{nd}f}%"


def _num(x, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def compute_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the pass/fail verdict from the numbers."""
    ct = result["cost_tax"]
    mid = next((r for r in ct if abs(r["cost_per_side"] - 0.0020) < 1e-9), ct[0])
    strat = result["strategy_perf"]
    bench = result["benchmark_perf"]["pre_tax"]
    after_tax_edge = mid["after_tax_annual"]
    strat_sharpe = strat.get("sharpe")
    bench_sharpe = bench.get("sharpe")

    beats_bench = (
        strat_sharpe is not None and bench_sharpe is not None
        and math.isfinite(strat_sharpe) and math.isfinite(bench_sharpe)
        and strat_sharpe > bench_sharpe
    )
    positive_after_tax = after_tax_edge > 0.0
    down = result["regime"]["down_tape"]
    has_bear_evidence = bool(down.get("n")) and (down.get("mean_net") or 0) > 0

    survives = bool(positive_after_tax and beats_bench)
    return {
        "survives": survives,
        "after_tax_annual_edge": after_tax_edge,
        "beats_benchmark_riskadj": beats_bench,
        "strat_sharpe": strat_sharpe,
        "bench_sharpe": bench_sharpe,
        "has_bear_evidence": has_bear_evidence,
        "fill_decay_annual": result["edge"]["fill_decay"] * 52,
    }


def write_report(result: Dict[str, Any], out: Path,
                 *, tax_rate: float = 0.35) -> Path:
    v = compute_verdict(result)
    e = result["edge"]
    L: list[str] = []
    A = L.append

    # ---- verdict (top) ----
    A("# VALIDATION.md — edge validity (paper only)\n")
    A(f"*Tickers: {', '.join(result['tickers'])} · benchmark {result['benchmark']} "
      f"· {result['n_trades']} walk-forward trades · tax {tax_rate:.0%} (short-term).*\n")
    A("## VERDICT\n")
    if v["survives"]:
        A(f"**The edge SURVIVES the honest tests** (provisionally). After "
          f"executable next-open fills, 0.20%/side round-trip cost, and a "
          f"{tax_rate:.0%} short-term tax drag, the annualized after-tax edge is "
          f"**{_pct(v['after_tax_annual_edge'])}**, and the strategy's Sharpe "
          f"({_num(v['strat_sharpe'])}) exceeds {result['benchmark']} buy-hold "
          f"({_num(v['bench_sharpe'])}). This is necessary but NOT sufficient: "
          f"note the fill decay, the bear-market evidence, and the sizing "
          f"section before risking capital.\n")
    else:
        A(f"**The edge does NOT survive.** After executable next-open fills, "
          f"0.20%/side cost, and a {tax_rate:.0%} short-term tax drag, the "
          f"annualized after-tax edge is **{_pct(v['after_tax_annual_edge'])}** "
          f"and the strategy's risk-adjusted return (Sharpe "
          f"{_num(v['strat_sharpe'])}) does {'' if v['beats_benchmark_riskadj'] else 'NOT '}"
          f"beat {result['benchmark']} buy-hold (Sharpe {_num(v['bench_sharpe'])}). "
          f"**Recommendation: do not deploy.** Trading this costs money and "
          f"underperforms simply holding the index. The apparent edge in the "
          f"paper log is an artifact of clean-close fills and untaxed, "
          f"cost-free settlement.\n")

    # ---- item 1 ----
    A("## 1 — Executable fills\n")
    A("| Fill | Mean/trade | Annualized |")
    A("|---|---|---|")
    A(f"| Signal-bar close | {_pct(e['signal_close_mean'])} | {_pct(e['signal_close_annual'])} |")
    A(f"| Next executable open | {_pct(e['executable_mean'])} | {_pct(e['executable_annual'])} |")
    A(f"\nFill decay (close − executable): **{_pct(e['fill_decay'])}/trade "
      f"≈ {_pct(v['fill_decay_annual'])}/yr**. This is the return the paper "
      f"log claims but the bot cannot capture.\n")

    # ---- item 2 ----
    A("## 2 — Full cost + tax model (executable fills)\n")
    A("| Cost/side | Round trip | Gross ann. | Net ann. | After-tax ann. |")
    A("|---|---|---|---|---|")
    for r in result["cost_tax"]:
        A(f"| {r['cost_per_side']*100:.2f}% | {r['cost_per_side']*200:.2f}% | "
          f"{_pct(r['gross_annual'])} | {_pct(r['net_annual'])} | "
          f"{_pct(r['after_tax_annual'])} |")
    A("")

    # ---- item 3 ----
    A("## 3 — Benchmark, risk-adjusted, after-tax\n")
    s, b = result["strategy_perf"], result["benchmark_perf"]
    bp = b["pre_tax"]
    A("| Metric | Strategy (net+tax) | " + result["benchmark"] + " buy-hold (pre-tax) |")
    A("|---|---|---|")
    A(f"| Sharpe | {_num(s['sharpe'])} | {_num(bp['sharpe'])} |")
    A(f"| Sortino | {_num(s['sortino'])} | {_num(bp['sortino'])} |")
    A(f"| Max drawdown | {_pct(s['max_drawdown'])} | {_pct(bp['max_drawdown'])} |")
    A(f"| CAGR | {_pct(s['cagr'])} | {_pct(bp['cagr'])} |")
    A(f"\n{result['benchmark']} after a single terminal long-term cap-gains tax: "
      f"CAGR {_pct(b['after_tax_cagr'])} (buy-hold defers tax until sale — the "
      f"weekly strategy pays the ordinary rate every trade). "
      f"**One sentence: after costs and taxes, the strategy "
      f"{'DOES' if v['beats_benchmark_riskadj'] else 'does NOT'} beat holding "
      f"{result['benchmark']} on a risk-adjusted basis.**\n")

    # ---- item 4 ----
    A("## 4 — Regime / downside evidence\n")
    A("| Year | Trades | Mean net/trade | Hit rate | Year total |")
    A("|---|---|---|---|---|")
    for y, row in result["regime"]["by_year"].items():
        A(f"| {y} | {row['n']} | {_pct(row['mean_net'])} | "
          f"{row['hit_rate']*100:.0f}% | {_pct(row['total'])} |")
    d = result["regime"]["down_tape"]
    A(f"\n**Bear tape** ({result['benchmark']} drawdown > 10%): "
      f"{d['n']} trades, mean net {_pct(d['mean_net'])}, "
      f"hit rate {(_num(d['hit_rate']*100,0)+'%') if d['hit_rate'] is not None else 'n/a'}, "
      f"total {_pct(d['total'])}.")
    if not d["n"]:
        A("\n**There is essentially NO bear-market evidence** — the strategy "
          "barely trades in downturns, so the entire edge is a bull-phase "
          "phenomenon. Treat any positive headline number as untested where "
          "it matters most.\n")
    else:
        A("")

    # ---- item 5 ----
    A("## 5 — Statistical honesty\n")
    lc = result["lens_corr"]
    A(f"**(a) Lens independence.** EMA/RSI-trend vs 12-1 momentum: "
      f"Pearson **{_num(lc.get('pearson'))}**, sign-agreement "
      f"**{(_num(lc.get('sign_agreement')*100,0)+'%') if lc.get('sign_agreement') is not None else 'n/a'}**. "
      f"They are NOT independent — both are trend proxies.")
    al = result["agreement_lift"]
    A(f"Agreement filter marginal lift over one lens: mean "
      f"{_pct(al.get('marginal_mean_lift'))}, hit-rate "
      f"{(_num(al.get('marginal_hit_lift')*100,0)+'pp') if al.get('marginal_hit_lift') is not None else 'n/a'}. "
      f"If the lift is small, 'two witnesses' is really one witness wearing two hats.\n")
    en = result["effective_n"]
    A(f"**(b) Effective bets behind the 20-trade kill-switch.** Avg pairwise "
      f"correlation of concurrent names **{_num(en.get('rho_bar'))}** → "
      f"**N_effective ≈ {_num(en.get('n_eff'),1)}** independent bets, not 20. "
      f"The kill-switch is a comfort blanket at roughly "
      f"{_num(en.get('n_eff'),0)}-bet resolution.\n")
    k = result["kelly"]
    if "edge_point" in k:
        A(f"**(c) Edge CI + Kelly under uncertainty.** Per-trade edge "
          f"{_pct(k['edge_point'])} (95% CI {_pct(k['edge_ci_lo'])} … "
          f"{_pct(k['edge_ci_hi'])}). Kelly f* at the point estimate "
          f"{_num(k['f_star_point'])}, at the CI lower bound "
          f"{_num(k['f_star_ci_lo'])}, at half-edge {_num(k['f_star_half'])}, "
          f"at zero-edge 0. Deployed quarter-Kelly of the point estimate = "
          f"{_num(k['deployed_quarter_kelly'])}. **If the true edge is half the "
          f"estimate, current sizing over-bets by "
          f"{_num(k['overbet_factor_if_true_edge_half'],1)}×** — and if the CI "
          f"lower bound is ≤ 0, the honest Kelly fraction is 0 (do not bet).\n")

    A("## Out of scope (flagged, not built)\n")
    A("- **shorts→puts** is NOT implemented. A long put does not pay off "
      "linearly with the underlying (delta<1, theta, vega); mapping an "
      "equity-calibrated edge onto options is a silent category error. If "
      "options are ever pursued they need their own option-level calibration "
      "— a separate project.\n")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


_HORIZON_NAME = {5: "weekly", 21: "monthly", 63: "quarterly",
                 126: "semi-annual", 252: "annual (long-term tax)"}


def write_confirm_report(res: Dict[str, Any], out: Path) -> Path:
    """Render the one-horizon stability confirmation."""
    L: list[str] = []
    A = L.append
    if "error" in res:
        A("# CONFIRM — could not run\n")
        A(res["error"])
        out.write_text("\n".join(L) + "\n", encoding="utf-8")
        return out
    name = _HORIZON_NAME.get(res["horizon_days"], f"{res['horizon_days']}d")
    sig = res.get("signal", "trend")
    sig_label = {"trend": "trend/momentum", "reversal": "oversold-reversal"}.get(
        sig, sig)
    A(f"# CONFIRM — is the {sig_label} {name} edge stable, or one lucky stretch?\n")
    A(f"*Tickers: {', '.join(res['tickers'])} · benchmark {res['benchmark']} · "
      f"cost {res['cost_side']*100:.2f}%/side · tax {res['tax_rate']:.0%} · "
      f"history split at {res['split_date']}.*\n")

    A("## VERDICT\n")
    if res["stable"]:
        A(f"**STABLE (necessary, not sufficient).** The {name} strategy beats "
          f"{res['benchmark']} risk-adjusted in BOTH halves of history "
          f"independently. That survives the 'one lucky stretch' objection — "
          f"but both halves are still the same bull decade, so the honest "
          f"next gate is live paper trading at this cadence, not real money.\n")
    else:
        which = ("first" if not res["first_half"]["beats"] else "second")
        A(f"**NOT STABLE.** The {name} strategy fails to beat "
          f"{res['benchmark']} in the {which} half of history. The sweep's "
          f"'winner' was carried by one stretch — treat it as luck, not edge. "
          f"**Do not fund it.**\n")

    def half_block(title: str, hb: Dict[str, Any]) -> None:
        s, b = hb["strategy"], hb["benchmark"]
        A(f"## {title}\n")
        A("| Metric | Strategy | " + res["benchmark"] + " |")
        A("|---|---|---|")
        A(f"| Sharpe | {_num(s['sharpe'])} | {_num(b['sharpe'])} |")
        A(f"| CAGR | {_pct(s['cagr'])} | {_pct(b['cagr'])} |")
        A(f"| Max drawdown | {_pct(s['max_drawdown'])} | {_pct(b['max_drawdown'])} |")
        A(f"| Beats benchmark? | **{'YES' if hb['beats'] else 'no'}** | — |")
        wr = hb.get("win_rate")
        aw, al = hb.get("avg_win"), hb.get("avg_loss")
        A(f"\nTrades {hb['n_trades']} · win rate "
          f"{(str(round(wr*100)) + '%') if wr is not None else 'n/a'} · "
          f"avg win {_pct(aw)} · avg loss {_pct(al)} · worst losses: "
          + (", ".join(_pct(x) for x in hb.get("worst_losses") or []) or "none")
          + "\n")

    half_block(f"First half (→ {res['split_date']})", res["first_half"])
    half_block(f"Second half ({res['split_date']} →)", res["second_half"])
    half_block("Overall", res["overall"])

    A("## What this does and does not prove\n")
    A("- Passing means the edge was not carried by a single stretch of the "
      "2018–2026 window. It does NOT create bear-market evidence, and it does "
      "not remove the closet-index concern — the strategy owns the same "
      "trending mega-caps the benchmark owns.\n"
      "- The only remaining honest gate before small real money: **live paper "
      "trading at this cadence**, scored by the same settlement machinery, "
      "matching the backtest's predictions within reason.\n")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def write_turnover_report(sweep: Dict[str, Any], out: Path) -> Path:
    """Render the Option-B holding-horizon sweep. The verdict is computed:
    a horizon 'works' only if its after-tax edge > 0 AND it beats the
    benchmark's Sharpe. Multiple-testing risk is stated, not hidden."""
    L: list[str] = []
    A = L.append
    A("# VALIDATION_TURNOVER.md — does trading LESS often survive? (paper only)\n")
    A(f"*Tickers: {', '.join(sweep['tickers'])} · benchmark {sweep['benchmark']} "
      f"(Sharpe {_num(sweep['benchmark_sharpe'])}, CAGR "
      f"{_pct(sweep['benchmark_cagr'])}) · cost {sweep['cost_side']*100:.2f}%/side · "
      f"short-term tax {sweep['short_tax']:.0%}, long-term {sweep['long_tax']:.0%}.*\n")

    survivors = [r for r in sweep["rows"]
                 if r["after_tax_annual"] > 0 and r["beats_benchmark_riskadj"]]

    A("## VERDICT\n")
    if not survivors:
        A("**No holding horizon survives.** Trading less often reduces the cost "
          "and tax drag, but at no tested horizon does the after-tax edge turn "
          "positive AND beat the benchmark on a risk-adjusted basis. The "
          "signal's pre-cost edge is real but too small to clear real-world "
          "frictions at any turnover. **Recommendation: do not deploy at any "
          "frequency — hold the index.**\n")
    else:
        names = ", ".join(f"{_HORIZON_NAME.get(r['horizon_days'], str(r['horizon_days'])+'d')}"
                          for r in survivors)
        A(f"**{len(survivors)} of {sweep['n_horizons_tested']} tested horizons "
          f"clear the bar** ({names}): after-tax edge > 0 AND Sharpe beats "
          f"{sweep['benchmark']}. **This is a lead, NOT a green light** — see "
          f"the multiple-testing caveat below before believing it.\n")

    A("## Horizon sweep\n")
    A("| Hold | Trades/yr | N | Tax | Gross ann. | After cost | After cost+tax | "
      "Sharpe | Beats bench? |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in sweep["rows"]:
        name = _HORIZON_NAME.get(r["horizon_days"], f"{r['horizon_days']}d")
        A(f"| {name} ({r['horizon_days']}d) | {r['trades_per_year']:.0f} | "
          f"{r['n_trades']} | {r['tax_rate']:.0%} | {_pct(r['gross_annual'])} | "
          f"{_pct(r['net_annual'])} | {_pct(r['after_tax_annual'])} | "
          f"{_num(r['sharpe'])} | {'YES' if r['beats_benchmark_riskadj'] else 'no'} |")
    A("")

    A("## The honest caveats (read before acting)\n")
    A(f"1. **Multiple testing.** {sweep['n_horizons_tested']} horizons were "
      "tried. If one squeaks past, that can be luck from taking several shots "
      "at the same data — the same trap the walk-forward is meant to avoid. A "
      "result is only believable if it wins by a **clear margin**, not a "
      "knife-edge, and ideally at more than one adjacent horizon.")
    A("2. **Fewer trades = weaker statistics.** A yearly hold over 8 years is "
      "~8 non-overlapping bets per name. 'It beat the benchmark' on a handful "
      "of bets is barely distinguishable from chance — check the trade count "
      "(N) column and distrust any row with a small N.")
    A("3. **Longer holds ≈ closet index.** As the hold lengthens the strategy "
      "increasingly just owns the same trending names the benchmark owns; a "
      "narrow win may be leverage/selection noise, not a distinct edge.")
    A("4. **Still no bear evidence.** Longer holds don't create downturn "
      "history that isn't in the 2018–2026 window. A horizon that 'wins' here "
      "is still untested in a sustained bear market.\n")

    A("## What a real 'yes' would require (not done here)\n")
    A("- The winning horizon holds up on a **held-out later slice** it was not "
      "chosen on.\n- It wins by a margin that survives a multiple-testing "
      "correction (e.g. Bonferroni across the horizons tried).\n- The trade "
      "count is large enough that the Sharpe has a tight confidence interval.\n"
      "Until then, treat any survivor as a hypothesis to test further, not a "
      "system to fund.\n")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
