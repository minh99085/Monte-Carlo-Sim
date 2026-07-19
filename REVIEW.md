# Engine Review: Monte-Carlo-Sim vs. QuantLib — and a Refactor Plan

**Target:** `minh99085/Monte-Carlo-Sim` (`mc_core.py`, 2,422 lines)
**Reference:** QuantLib (`ql/methods/montecarlo/`, `ql/processes/`, `ql/math/statistics/`)

QuantLib was read for *architecture only* — no code was translated or copied
(different language, different license). Everything proposed below is a
NumPy-native, chunk-vectorized reinterpretation of its layering.

---

## a. Side-by-side architecture

### How Monte-Carlo-Sim works today

`mc_core.py` is one module with four tightly coupled responsibilities:

1. **Configuration.** One `SimulationConfig` dataclass carries the union of
   every model's parameters (GBM, Student-t, two bootstraps, Merton, Kou,
   regime switching, Heston, GARCH): ~35 fields plus stress/VR/ruin knobs,
   with a 60-line `validate()` that dispatches per model.
2. **Model dispatch.** `_build_step_engine(cfg)` (~225 lines) is a chain of
   `if model == ...:` blocks. Each returns two closures:
   `init_chunk(rng, n) -> state` (chunk-local state, e.g. Heston's variance
   vector or the regime index array) and `step(rng, n, state, i) -> log_ret`
   (one vectorized log-return per path for step *i*). This is the engine's
   best idea — per-step, per-chunk closures already keep every model
   memory-bounded — but the closures fuse *drift*, *diffusion*,
   *shock drawing*, *variance-reduction plumbing* (`gauss()` peeks into
   `state["sobol_z"]`), and *discretization* into one lambda each.
3. **The loop.** `simulate(cfg)` (~210 lines) owns everything else:
   chunk slicing, Sobol shock attachment, the stress crash, per-step price
   updates, **and the payoff logic inlined into the loop** — running max,
   max drawdown, underwater streaks, ruin flags, the sample-trajectory
   block, terminal values, costs. Then the control-variate post-process,
   convergence curve, and the stats dict.
4. **Statistics/exports.** `compute_statistics()` runs *post hoc* on the
   full `final_values` array; report/CSV/JSON, model comparison, EVT tail
   fit, Kupiec/VaR backtests and a separate `simulate_portfolio()` hang off
   it.

**Structural constraints this creates:**

- **The payoff is hardcoded.** The only "instrument" the engine can price is
  *net terminal value of one asset*. Drawdown/ruin are the only
  path-dependent quantities, and they exist as loop-local accumulators —
  adding an Asian or barrier payoff means editing `simulate()` itself.
- **Single asset everywhere.** `simulate_portfolio()` is a second,
  independent implementation (GBM-only, correlated normals) rather than a
  multi-dimensional process fed to the same loop.
- **Variance reduction is entangled.** Antithetic lives in `_draw_gauss`,
  Sobol in a state-dict side channel that only some models honor
  (Student-t, Merton, regime silently ignore it), and the control variate is
  a special-cased post-process that only works for unstressed GBM.
- **No reusable "evolve" contract.** Because each model is a closure, nothing
  else can ask a model for its drift or diffusion — so Greeks, bridges,
  alternative discretizations, or an LSMC exercise rule have no seam to
  attach to.
- **What is genuinely good and must be preserved:** the chunked driver
  (`MemoryInfo` proves the full `paths × steps` matrix is never allocated),
  the per-chunk state discipline, reproducible seeding, the model breadth,
  and the honest fallbacks (Sobol → plain MC with a recorded note).

### How QuantLib decomposes the same problem

| Concern | QuantLib component | Contract |
|---|---|---|
| Dynamics | `StochasticProcess` / `StochasticProcess1D` | *Only* `drift(t, x)`, `diffusion(t, x)`, plus `expectation/stdDeviation/covariance` and `evolve(t0, x0, dt, dw)`; `size()`/`factors()` declare dimensionality |
| Discretization | nested `discretization` strategy object | Euler etc. are swappable *without touching the process*; e.g. `HestonProcess` ships nine schemes (full/partial truncation, reflection, Quadratic-Exponential, Broadie-Kaya) selected by an enum, applied inside an overridden `evolve` |
| Path construction | `PathGenerator<GSG>` / `MultiPathGenerator` | Owns the `TimeGrid` and the random *sequence generator*; calls `process->evolve(...)` per step; `next()`/`antithetic()` return `Sample<Path>`; an optional flag routes shocks through `BrownianBridge` |
| Payoff | `PathPricer<PathType>` | A callable `path -> Real`. The instrument is *data given to the loop*, not code inside it |
| Orchestration | `MonteCarloModel<MC, RNG, S>` | Glues generator + pricer + statistics; implements antithetic averaging and the control variate as *generic* wiring: `price + (cvValue − cvPricer(path))` for any pricer with a known-mean companion |
| Accumulation | `Statistics` / `GenericRiskStatistics` / `IncrementalStatistics` | Streaming accumulators: `add(value, weight)` during the run; moments, quantiles, VaR/ES, downside deviation queried afterwards |
| QMC | `BrownianBridge` (Jäckel's construction) + low-discrepancy RNG traits | Sobol dimensions are spent on the bridge's *coarse-to-fine* time ordering so the lowest (best) dimensions carry the most variance |
| Early exercise | `LongstaffSchwartzPathPricer` wrapping `EarlyExercisePathPricer` | Two-pass: calibration paths → backward ITM regression on `basisSystem()` → fitted continuation rule → fresh pricing paths |

The essence: **the loop knows nothing about models, payoffs, or statistics;
each is a small object with a one-method contract.** Swapping any axis
(model ↔ payoff ↔ RNG ↔ statistic) is free because they only meet at
`evolve`, `operator()`, and `add`.

### The one place we should *not* copy QuantLib

QuantLib prices **one path at a time** (a `Path` object per sample). In
NumPy that throughput model is untenable. The refactor below keeps the
target's *vectorized chunk* as the unit of work: a "path" is a chunk-sized
vector evolving step by step, and pricers consume it **streamingly**
(running sums/extrema/flags) so the full `chunk × steps` matrix is still
never required. This is the same layering as QuantLib with the axes
transposed — and it is exactly what makes Asian/barrier/lookback payoffs
possible *without* giving up the memory guarantee, because all three reduce
to per-step running statistics.

---

## b. Proposed refactor: decoupled layers

```
StochasticProcess  →  PathGenerator  →  PathPricer(s)  →  Statistics
 (drift+diffusion)     (chunk loop,       (streaming        (streaming
                        shocks, VR,        payoffs &         accumulators;
                        time grid)         observers)        VaR/ES/quantiles)
```

### Layer 1 — `StochasticProcess` (dynamics only)

```python
class StochasticProcess:
    """One asset (or state vector) evolving in log-price space."""
    def init_state(self, n: int) -> dict:          # chunk-local arrays, len n
        return {}
    def drift(self, state, n) -> np.ndarray | float:      # per-step log drift rate
        raise NotImplementedError
    def diffusion(self, state, n) -> np.ndarray | float:  # per-step vol
        raise NotImplementedError
    def evolve(self, rng, state, z, n) -> np.ndarray:     # log-return for the step
        # default Euler in log space; override for schemes that need more
        return self.drift(state, n) * self.dt + \
               self.diffusion(state, n) * self.sqrt_dt * z
```

Notes mirroring QuantLib deliberately:

- `drift`/`diffusion` are the *definition* of the process; `evolve` is the
  *discretization* and defaults to Euler. Processes with bespoke schemes
  (Heston full truncation today; QE later) override `evolve` only.
- `z` is the process's **primary Gaussian shock**, supplied by the
  generator (so antithetic/Sobol/bridge policy lives in one place).
  Processes that need extra randomness (Heston's second factor, Poisson
  jump counts) draw it from `rng` inside `evolve` and declare it via a
  `factors` attribute, exactly like QuantLib's `factors()`.
- State is a dict of chunk-length arrays — the existing `init_chunk`
  discipline, formalized.

**Every existing model collapses into a small subclass** (the arithmetic is
lifted verbatim from today's closures, so behavior is unchanged):

| Model | Subclass sketch | State |
|---|---|---|
| GBM | `drift = μ − σ²/2`, `diffusion = σ` (constants) | — |
| Student-t | override `evolve`: `t`-draw scaled by `√((ν−2)/ν)` | — |
| Historical bootstrap | override `evolve`: sample centered returns + target drift | — |
| Block bootstrap | override `evolve`: block cursor/remaining arrays | `cur`, `rem` |
| Merton | override `evolve`: Euler part + compound-Poisson normal jumps; compensator in `drift` | — |
| Kou | same shape, double-exponential jump magnitudes | — |
| Regime switching | `drift`/`diffusion` index per-regime vectors; Markov update in `evolve` | `regime` |
| Heston | `diffusion = √v⁺`; full-truncation variance update in `evolve` | `v` |
| GARCH(1,1) | `diffusion = √var`; recursion on realized shock in `evolve` | `var` |

The 225-line `if/elif` becomes a registry:
`PROCESS_REGISTRY = {MODEL_GBM: GBMProcess, ...}` +
`process_from_config(cfg)`. `SimulationConfig` keeps working as-is (the
factory reads the same fields), and per-model parameters can later migrate
into per-process dataclasses without breaking the flat config.

### Layer 2 — `PathGenerator` (the chunk driver)

```python
class PathGenerator:
    def __init__(self, process, *, s0, paths, steps, dt, chunk_size,
                 seed=None, antithetic=False, sobol=False, bridge=False): ...
    def run(self, pricers: Sequence[PathPricer]) -> None:
        for chunk in self._chunks():                  # bounded by chunk_size
            prices = np.full(n, s0); state = process.init_state(n)
            for p in pricers: p.begin_chunk(prices)
            for i in 1..steps:
                z = self._shock(i)                    # antithetic / Sobol / bridge
                prices *= np.exp(process.evolve(rng, state, z, n))
                for p in pricers: p.observe(i, prices)
            for p in pricers: p.end_chunk(prices)
```

This is today's `simulate()` while-loop with the model call and the payoff
accumulation replaced by two interfaces. Chunking, seeding, per-chunk Sobol
seeds, the stress crash overlay, and `MemoryInfo` accounting move here
unchanged — the memory guarantee is structural (nothing above ever holds
more than `chunk` floats per array, plus the bounded sample block).

### Layer 3 — `PathPricer` (streaming payoffs and observers)

```python
class PathPricer:                     # vectorized, streaming
    def begin_chunk(self, prices0): ...
    def observe(self, step_i, prices): ...   # optional for terminal payoffs
    def end_chunk(self, prices): ...
    def values(self) -> np.ndarray: ...      # one number per path
```

QuantLib's `pricer(path) -> Real` becomes "pricer consumes the chunk
step-by-step and emits a value per path". Terminal payoffs skip `observe`
entirely. Today's inlined drawdown/ruin/underwater block becomes a
`DrawdownObserver` — same math, now a reusable object; the sample-trajectory
block becomes a `SampleRecorder`.

### Layer 4 — streaming `Statistics`

QuantLib-style accumulators (`add(values, weights)` per chunk; mean/M2 via
Welford, P² or reservoir quantiles) so headline stats no longer *require*
the full `final_values` array. Near term the array stays (VaR/percentiles
and existing exports read it, and 5M float64 ≈ 40 MB is acceptable);
streaming becomes the default only for future >10⁷-path tail studies.
`compute_statistics`/`GenericRiskStatistics`-style metrics stay exactly
where they are — they already match QuantLib's separation.

---

## c. What this unlocks (proposed APIs and layout)

### File layout (flat, matching the repo's style)

```
mc_engine.py          # Phase 1: process + generator + pricer + streaming moments
mc_processes.py       # Phase 2+: remaining model ports (registry)
mc_pricers.py         # Phase 3+: Asian/barrier/lookback, DrawdownObserver
mc_greeks.py          # Phase 4: pathwise + likelihood-ratio Greeks (CRN)
mc_lsmc.py            # Phase 5: Longstaff-Schwartz early exercise
mc_qmc.py             # Phase 6: Brownian bridge + Sobol dimension policy
mc_kernels.py         # Phase 7: optional Numba inner loops, pure-NumPy fallback
mc_core.py            # façade: SimulationConfig, simulate(), stats, exports
```

### 1. Path-dependent payoffs — all streaming, all chunk-safe

```python
AsianPricer(strike, call=True)        # observe: running sum → arithmetic mean
BarrierPricer(barrier, knock="out")   # observe: crossing flag |= prices >= B
LookbackPricer(kind="floating")       # observe: running min/max
```

Each is ~15 lines because the generator already delivers per-step prices.
None allocates `chunk × steps`.

### 2. Longstaff-Schwartz early exercise

The one payoff family that genuinely needs stored paths. Following
QuantLib's two-pass design: a bounded **calibration set** (e.g. 20k paths —
explicitly allocated and reported in `MemoryInfo`, still never
`paths × steps` for the full run) fits the backward ITM regression on a
polynomial `basis(state)`; the pricing pass then streams normally, applying
the fitted continuation rule per step.
`AmericanExercisePricer(payoff, basis=poly(2), calib_paths=20_000)`.

### 3. Greeks with common random numbers

The process seam makes both standard estimators possible:

```python
greeks(cfg, pricer, bumps=("delta", "vega"))   # CRN: same seed, bumped process
PathwiseDelta(pricer)                           # GBM-family: ∂S_T/∂S₀ = S_T/S₀ streaming
LikelihoodRatioDelta(pricer)                    # score-function weight from z-draws
```

CRN re-runs are cheap because `PathGenerator` reproduces identical shock
streams per chunk from the seed — bumped and base runs share every draw.

### 4. Proper QMC: Sobol + Brownian bridge

Today Sobol spends dimension *i* on time step *i*, which wastes the
low-discrepancy sequence's best dimensions on fine detail. A
`BrownianBridge` transform (Jäckel's coarse-to-fine construction: terminal
point first, then midpoints) lets dimension 1 carry the terminal variance —
this is where QMC's convergence gain actually comes from — and applies
per chunk inside `_shock()`, invisible to processes and pricers. Also fixes
the current silent inconsistency where only GBM/Heston-z1/GARCH honor Sobol:
the generator owns shocks, so *every* Gaussian-driven process gets QMC.

### 5. Optional Numba/Cython inner loop

The `evolve`-per-step NumPy pipeline pays Python-loop overhead ∝ steps, not
paths — already fine. For the hot single-trade engines (see
`tactical_simulator._classic_stop_hold_kernel`, which already ships a Numba
kernel with a fallback) the same pattern extends to `mc_kernels.py`:
`@njit` chunk loops for GBM/Heston selected at import time, pure-NumPy
fallback guaranteed, identical arithmetic asserted by the equivalence tests.
CPU-only, no new required dependency.

---

## d. Phased migration plan (strangler fig — every phase ships green)

Ground rules for *every* phase: `python -m pytest -q` passes; CLI flags,
Streamlit app, JSON/CSV schemas and default outputs unchanged; new engine
is opt-in until its equivalence tests have soaked.

| Phase | Scope | Risk kept low by |
|---|---|---|
| **1 (this PR)** | `mc_engine.py` with `StochasticProcess`, `PathGenerator`, `PathPricer`, streaming moments; **GBM + Heston** ported; opt-in via `SimulationConfig.engine="v2"` or `MC_ENGINE=v2`; legacy default untouched | Adapter reuses the legacy chunk loop's exact draw order → **byte-identical** outputs, asserted by tests |
| **2** | Port the remaining 7 models onto `StochasticProcess`; `_build_step_engine` becomes registry + thin adapter; flip the default to v2; keep `engine="legacy"` as escape hatch for one release | Per-model exact-equality tests (same seed) before each port flips |
| **3** | Extract `DrawdownObserver`/`SampleRecorder` from `simulate()`; add Asian/barrier/lookback pricers + CLI (`--payoff asian --strike ...`) as *new* flags | Old outputs are produced by the same observers, just relocated; golden-file JSON diff test |
| **4** | Greeks: CRN bump harness + pathwise/LR estimators; verify against Black-Scholes closed forms for GBM | Pure addition; analytic oracles exist |
| **5** | Longstaff-Schwartz American pricer (calibration/pricing split per QuantLib) | Validate vs binomial-tree prices on test cases |
| **6** | Brownian-bridge Sobol ordering; generator-owned shocks for all models; deprecate the `state["sobol_z"]` side channel | Convergence-rate test (QMC error < MC error at equal paths) plus fallback note preserved |
| **7** | Optional Numba kernels for GBM/Heston inner loops; `simulate_portfolio` re-expressed as a multi-dimensional process on the same generator | Kernel-vs-NumPy exact equality; portfolio golden files |

Rollback at any phase = flip the engine default back; no phase deletes
legacy code until the phase after its equivalence tests have been the
default.

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 1 — process/generator/pricer abstractions, GBM+Heston behind flag | **Done** (commit 4af95dd) | bit-identical to legacy; 23 tests |
| 2 — payoff-agnostic engine: path-dependent pricers + Longstaff-Schwartz | **Done** (commit f43cf66) | see validation below; 39 tests |
| 3 — Greeks (PW / LRM / CRN-FD) + Sobol Brownian bridge | **Done** (this change) | see validation below; 27 tests |
| 4+ — remaining model ports, observers, kernels | pending | plan unchanged (§d) |

**Deviations from the original §c/§d plan, by request:** Phase 2 was
re-scoped from "port the remaining 7 models" to "prove the PathPricer
abstraction" (originally Phases 3 and 5); the model ports move to Phase 3+.
File names differ from the §c sketch: `mc_payoffs.py` (was `mc_pricers.py`)
and `mc_lsm.py` (was `mc_lsmc.py`); `mc_options.py` is the options entry
point. The LSM pricer stores a documented `paths × K` exercise-date matrix
(assembled chunk-by-chunk, calibration matrix freed before the pricing
pass) — the one agreed exception to full streaming; everything else
(`Asian`, barrier, lookback) is running-statistic streaming as designed.

### Phase 2 validation results (risk-neutral GBM, S0=100, K=100, r=5%, σ=25%, T=1y, 252 steps, 100k paths, seed 42)

| Check | MC price (se) | Reference | Verdict |
|---|---|---|---|
| European call vs Black-Scholes | 12.3729 (0.0586) | 12.3360 | within 0.63 se ✔ |
| Down-and-out call, B=85, vs Reiner-Rubinstein @ BGK-adjusted barrier | 11.2690 (0.0585) | 11.2447 | within 0.42 se ✔; above the unadjusted continuous formula 11.0529, confirming the discrete-monitoring bias direction |
| Geometric Asian vs exact discrete closed form (Kemna-Vorst) | 6.5431 (0.0302) | 6.5517 | within 0.28 se ✔; arithmetic Asian 6.8663 dominates path-by-path (AM-GM) and sits below the European |
| Antithetic variance reduction (European, 40k paths) | paired se 0.0687 | plain se 0.0927 | 26% se reduction ✔ (Sobol beats plain MC on mean abs error across seeds — tested) |
| LSM American put 36/0.2/1y (20k+20k paths, K=50 dates, degree 3) | 4.4851 (0.0199) | LS2001 4.472; CRR tree 4.4867 | within 0.7 se of the paper and 0.1 se of the tree ✔ |
| LSM grid (S0∈{36,40,44}, σ∈{0.2,0.4}, T∈{1,2}) | — | LS2001 Table 1 + in-test CRR oracle | all within max(0.08, 4 se) ✔ |
| American call, no dividends | 4.3772 | BS European 4.3958 | equal within tolerance ✔ (early exercise ≈ never optimal) |

Chunk-invariance is asserted three ways (pricer state machines are exactly
invariant on identical paths under different chunk splits; a zero-vol
process is exactly invariant end-to-end; the random pipeline is invariant
within MC error — bit-exact invariance across chunk sizes is impossible
with a shared sequential PRNG stream, same as the legacy engine).

### Phase 3 validation — Part A: Greeks (risk-neutral GBM, S0=K=100, r=5%, σ=25%, T=1y, 100k paths, seed 42)

| Check | Estimate (SE) | Closed form | Verdict |
|---|---|---|---|
| Pathwise delta (European) | +0.629115 (0.001924) | N(d1) = 0.627409 | 0.89 SE ✔ |
| Pathwise vega (European) | +37.9881 (0.2531) | S0·n(d1)·√T = 37.8420 | 0.58 SE ✔ |
| LRM delta (European) | +0.629156 (0.004742) | 0.627409 | 0.37 SE ✔ — agrees with PW within combined SE; SE 2.5× PW, as documented |
| LRM vega (European) | +37.7589 (0.8774) | 37.8420 | 0.09 SE ✔ |
| LRM gamma (European) — where PW is refused | +0.015104 (0.000351) | n(d1)/(S0σ√T) = 0.015137 | 0.09 SE ✔ |
| Digital call: PW delta | **rejected** (`MethodValidityError` steering to LRM/FD) | — | validity logic ✔ |
| Digital call: LRM delta | +0.015195 (0.000070) | e^{-rT}n(d2)/(S0σ√T) = 0.015137 | 0.83 SE ✔ |
| CRN FD delta | +0.629329 (0.001911) | independent-seed FD: +0.648500 (0.041401) | SE ratio 21.7× ⇒ variance ratio ≈ **470×** collapsed by CRN ✔ |
| Asian: PW delta / vega vs CRN-FD cross-check | 0.579821 / 22.2714 | FD-CRN 0.5798 / 22.2698 | agree to 4 decimals ✔ |

Method-validity matrix enforced by the dispatcher: PW = delta/vega on
smooth payoffs only (gamma and digitals **raise**); LRM = delta/vega/gamma
on terminal payoffs (exact GBM lognormal density; per-step scores for
path-dependent payoffs are future work); CRN-FD = universal cross-check
(any pricer, any process incl. Heston), bias O(bump²).

### Phase 3 validation — Part B: Sobol + Brownian bridge (GBM, European S0=K=100, 32 steps; Asian 64 steps)

RMS pricing error over 16 scrambled-seed replications:

| N (paths) | PRNG | Sobol + bridge |
|---|---|---|
| 256 | 1.1273 | 0.1169 |
| 1,024 | 0.7130 | 0.0230 |
| 4,096 | 0.3176 | 0.0053 |
| 16,384 | 0.0938 | 0.0009 |
| **fitted slope** | **−0.596** (≈ N^−1/2) | **−1.155** (≈ N^−1) |

Headline (path-dependent payoff, N = 4,096, 64 steps — RMS error):

| Payoff | PRNG | plain Sobol (per-step dims) | Sobol + bridge |
|---|---|---|---|
| Geometric Asian (exact discrete closed form as reference) | 0.2002 | 0.0385 | **0.0047** (8.2× better than plain Sobol) |
| Arithmetic Asian (2^18-path reference 6.9382) | 0.1314 | 0.0202 | **0.0055** (3.7× better than plain Sobol) |

The bridge assigns Sobol dimension 1 to the terminal Brownian level and
successive dimensions to recursive midpoints (Jäckel's construction,
reimplemented in NumPy), so the best-distributed dimensions carry the most
path variance. Scope note: the bridge is single-factor only this phase —
`PathGenerator(bridge=True)` **raises** for multi-factor processes
(Heston's correlated variance shock would need a joint two-factor
dimension allocation); plain Sobol and pseudorandom modes are unchanged
and remain available for Heston. Greeks are likewise GBM-analytic for
PW/LRM (the weights use the GBM density); CRN-FD covers Heston.

## Phase 1 implementation (commit 4af95dd)

Delivered in `mc_engine.py` + `test_mc_engine.py`:

- `StochasticProcess` base (drift/diffusion contract, default Euler
  `evolve`, chunk-local `init_state`), `GBMProcess`, `HestonProcess`
  (full-truncation `evolve` override — arithmetic identical to the legacy
  closure), and `process_from_config()`.
- `PathGenerator`: chunk-safe driver with the same seeding, antithetic,
  per-chunk Sobol and stress-crash semantics as `simulate()`; drives
  streaming `PathPricer`s; tracks a `MemoryInfo`-compatible peak so the
  chunk guarantee is testable.
- `PathPricer` base + `TerminalValuePricer` (net-of-cost terminal values via
  the same `apply_costs`), plus `StreamingMoments` (Welford) as the seed of
  the statistics layer.
- Feature flag: `SimulationConfig.engine = "legacy"` (default) — `"v2"`
  routes GBM/Heston through the process objects inside the existing loop
  (byte-identical results); `MC_ENGINE=v2` env var works without CLI
  changes. Unsupported models fall back to legacy closures.
- Tests: bit-exact equality legacy↔v2 for GBM and Heston (plain,
  antithetic, Sobol, stress), statistical agreement of the standalone
  pipeline within Monte Carlo tolerance, chunk-safety assertions, and
  process-contract unit tests.
