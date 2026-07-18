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

## Phase 1 implementation (in this change)

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
