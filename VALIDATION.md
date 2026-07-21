# VALIDATION.md — edge validity (paper only)

> **STATUS: NOT YET RUN ON REAL MARKET DATA — verdict UNRESOLVED.**
> The measurement instrument (`validate_edge.py`) is built and proven correct
> on synthetic ground truth (14 tests, `test_validate_edge.py`). The *real*
> SPY/QQQ/NVDA numbers cannot be produced in the build sandbox: its network
> policy blocks every market-data host (Yahoo, stooq — confirmed 403). The
> honest verdict for real tickers comes from **one command on the VPS**, where
> Yahoo is reachable:
>
> ```
> /opt/monte-carlo-sim/.venv/bin/python /opt/monte-carlo-sim/validate_edge.py \
>     --tickers SPY QQQ XLK NVDA AAPL MSFT --benchmark QQQ --years 8 \
>     --out /opt/monte-carlo-sim/VALIDATION.md
> ```
>
> That run **overwrites this file** with the real numbers and a
> computed-from-the-numbers verdict. I did not fabricate SPY figures to fill
> the gap — that would be the exact opposite of this task.

## What the current state actually means (this is itself a finding)

The live **paper log is not evidence of a real edge.** It settles every trade
at the clean daily close, with no spread, no slippage, no tax, and no
benchmark. Every one of those omissions flatters the result. Until the VPS run
above shows the edge clearing executable fills + full costs + tax on a
risk-adjusted basis versus QQQ, **the edge is UNVALIDATED and the system
should not trade real money.** "It's green in paper" is precisely the illusion
this exercise exists to puncture.

Three of the checks below do not need the live run to state their direction —
they are structural, and they are unfavorable.

---

## 1 — Executable fills *(needs VPS run for real numbers)*

The strategy decides on a daily close but a real order fills at the **next
open**. `walk_forward_trades()` records both, side by side, for every trade,
and reports the annualized **fill decay** = (signal-close edge − executable
edge). On a smooth synthetic series the decay is ~0 by construction; on real
data the Monday-gap is where a ~0.27%/week edge most plausibly dies. This is
the single most important number the VPS run will produce.

## 2 — Full cost + tax model *(needs VPS run)*

Round-trip cost applied multiplicatively at **0.10 / 0.20 / 0.40 % per side**,
annualized over the real trade frequency, then a **35% short-term** cap-gains
drag (parameter `--tax-rate`). Reported as gross → net → after-tax at each
cost level. Weekly churn is taxed at the ordinary rate on *every* winning
trade — the most punitive tax treatment there is.

## 3 — Benchmark, risk-adjusted, after-tax *(needs VPS run)*

Strategy (net + tax) vs QQQ **and** SPY buy-and-hold over the identical window
— Sharpe, Sortino, max drawdown, CAGR. The comparison is deliberately
asymmetric in the benchmark's favor on tax (buy-hold defers tax until sale;
the report taxes the benchmark's terminal gain once at the long-term rate),
because that asymmetry is real and the strategy must beat it anyway. The
report prints the one-sentence answer: *after costs and taxes, does this beat
holding the index on a risk-adjusted basis — yes/no.*

## 4 — Regime / downside evidence *(needs VPS run)*

Trade results segmented by calendar year, and isolated to windows where QQQ
drew down >10% (2022 explicitly). The report states plainly whether there is
**any** bear-market evidence or whether the entire edge is a 2023–2025 bull
phenomenon. Given the watchlist and the 2018–2026 window (mostly up), expect
few trades in downturns — i.e. expect "essentially no bear evidence." The VPS
run will confirm the exact counts.

## 5 — Statistical honesty

**(a) The two lenses are not independent — structural.** EMA/RSI-trend and
12-1 momentum are both trend proxies. The synthetic demo below shows Pearson
≈ 0 *only because a random walk has no trend to correlate on*; on real
trending names the correlation and sign-agreement will be **substantially
higher**, and the "agreement filter" therefore adds far less marginal
information than "two independent witnesses" implies. The VPS run prints the
real Pearson, sign-agreement, and the filter's actual marginal lift.

**(b) The 20-trade kill-switch is worth far fewer than 20 independent bets —
structural.** The watchlist (SPY, QQQ, XLK, NVDA, AAPL, MSFT …) is dominated
by one tech factor; realized pairwise correlations of these names run ~0.6–0.9.
With `N_eff = N / (1 + (N−1)·ρ̄)`, at ρ̄ = 0.7 twenty trades collapse to
**N_eff ≈ 1.4 independent bets**. The synthetic demo shows N_eff ≈ 20 *only
because the synthetic tickers are mutually independent GBM* — the real number
will be a small single digit. Do not trust the kill-switch as 20-trade
statistical protection.

**(c) Edge CI + Kelly under uncertainty.** The report bootstraps the per-trade
edge to a 95% CI and re-derives Kelly `f* ≈ mean/variance` at the point
estimate, the **CI lower bound**, half-edge, and zero-edge. The logic, proven
in tests: if the true edge is half the estimate, the deployed quarter-Kelly
**over-bets by exactly 2×**; if the CI lower bound is ≤ 0 (entirely plausible
for a ~0.27%/week edge with wide dispersion), the honest Kelly fraction is
**0 — do not bet.** Quarter-Kelly of an over-estimated edge is not the
conservative sizing it appears to be.

---

## Harness demonstration (SYNTHETIC — not real market data)

Proof the instrument works and renders an honest fail. This is a **zero-edge
GBM world** (three independent random-walk "tickers", a benchmark drifting
+10%/yr), run through the full pipeline. It is NOT SPY/NVDA/AAPL — the names
are placeholders — and it **understates** items 5a/5b for the reasons above.

```
VERDICT: The edge does NOT survive.
  After executable fills, 0.20%/side cost, 35% tax:
  after-tax annual edge  -35.53%
  strategy Sharpe -2.63  vs  QQQ buy-hold Sharpe +1.21
  strategy CAGR  -26.67% vs  QQQ CAGR (after-tax) +16.61%
  Recommendation: do not deploy.

  1  Fill decay (close - executable):  ~ -4.4%/yr on this synthetic series
  2  Net after-tax edge @0.10/0.20/0.40% side:  -26.3% / -35.5% / -54.1% /yr
  4  Bear tape (QQQ DD>10%): 18 trades, hit rate 28%, total -10.2%
  5a Lens Pearson 0.00 (≈0 ONLY because GBM has no trend; real data: higher)
  5b N_eff ≈ 20 (≈20 ONLY because synthetic names are independent; real: ~1-3)
  5c Edge -0.09%/trade, 95% CI [-0.55%, +0.39%] — CI includes 0 → Kelly 0
```

The zero-edge world correctly returns a large *negative* after-tax result — the
harness does not manufacture edge. A companion test plants a real conditional
drift and confirms the harness reports a positive edge when one genuinely
exists, so it is neither rigged to pass nor rigged to fail
(`test_validate_edge.py`).

---

## VERDICT (provisional, pending the VPS run)

On the evidence available without the live run, the responsible position is:
**treat the edge as UNVALIDATED and do not deploy real capital.** The paper
track record is not proof; two of the three statistical checks are
structurally unfavorable before a single real number is computed; and the one
measurement that could rescue the case — executable fills surviving full costs
and tax versus QQQ — has not been run. Run the VPS command, read the
computed verdict at the top of the regenerated file, and believe it even if it
says "does not survive."

## Part 3 (ops hardening) — DEFERRED, by the task's own gate

Hardening (secret rotation, HTTPS, dashboard auth, offsite backups, failure
alerting, exchange-calendar scheduling, yfinance fail-safe) is **not done in
this pass.** The task gates it on "Part 1 verdict is that a real edge
survives," and Part 1 is currently UNRESOLVED (and structurally leaning
negative). Hardening an unvalidated edge is premature — it makes an unproven
system more robust at being unproven. Revisit only if the VPS run shows the
edge genuinely survives. (The specific hardening items and their rationale are
recorded in `deploy/cohost/BOT_REVIEW.md` §Part 1 and the blindspot notes, so
nothing is lost by deferring.)

## Out of scope — flagged, not built

**shorts→puts** is deliberately NOT implemented. A long put does not pay off
linearly with the underlying (delta<1, theta decay, vega): a correctly
predicted −1%/week move can still *lose* money as a put. Mapping the
equity-calibrated edge onto options is a silent category error. If options are
ever pursued they require their **own option-level calibration** on option
P&L — a separate project, not a mapping.
