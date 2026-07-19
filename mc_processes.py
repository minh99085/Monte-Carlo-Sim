"""
mc_processes.py — Phase 4: the remaining seven legacy models as
StochasticProcess subclasses (REVIEW.md §b table; file name per §c layout).

Every ``evolve`` here is a line-for-line transcription of the corresponding
closure in ``mc_core._build_step_engine`` — same arithmetic, same expression
order, and (the hard constraint) the **same RNG call sequence** — so a
flagged ``engine="v2"`` run is byte-identical to the legacy engine
(``np.array_equal``), verified per model in ``test_mc_processes.py``.

Shock policy per model (see ``StochasticProcess.gauss_mode``):

===================  ==========  =============================================
model                gauss_mode  legacy randomness, in draw order
===================  ==========  =============================================
Student-t GBM        none        ``rng.standard_t(df, n)`` (scaled)
Historical Bootstrap none        ``rng.integers(0, n_hist, n)``
Block Bootstrap      none        ``rng.integers(0, n_hist, n)`` only on steps
                                 where any path starts a new block
Merton Jump          none        ``standard_normal(n)``, ``poisson(n)``,
                                 ``standard_normal(n)`` (jump-sum trick)
Kou Jump             plain       ``gauss(rng, n)`` (antithetic, never Sobol),
                                 ``poisson(n)``, then per-jump
                                 ``random(total)`` / two ``exponential`` draws
GARCH(1,1)           full        ``gauss(rng, n, state, i)`` (Sobol-eligible);
                                 variance recursion on the realized shock
Regime Switching     none        ``rng.random(n)`` (Markov transition) *then*
                                 ``rng.standard_normal(n)``
===================  ==========  =============================================

Not every model has a clean drift/diffusion split: the bootstraps resample
empirical returns (no Gaussian decomposition — ``drift``/``diffusion`` stay
unimplemented, which the base class permits when ``evolve`` is overridden),
and regime/GARCH expose state-dependent forms for completeness.

All state is chunk-local (regime index, block cursors, conditional
variance): memory-safe chunking holds for every model.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from mc_core import (
    MODEL_BLOCK_BOOTSTRAP,
    MODEL_GARCH,
    MODEL_HIST_BOOTSTRAP,
    MODEL_KOU,
    MODEL_MERTON,
    MODEL_REGIME,
    MODEL_STUDENT_T,
    REGIME_PRESETS,
    SimulationConfig,
    _bootstrap_daily_drift,
)
from mc_engine import StochasticProcess

__all__ = [
    "StudentTProcess",
    "HistoricalBootstrapProcess",
    "BlockBootstrapProcess",
    "MertonJumpProcess",
    "KouJumpProcess",
    "GARCHProcess",
    "RegimeSwitchingProcess",
    "extended_process_from_config",
]


class StudentTProcess(StochasticProcess):
    """GBM with Student-t shocks (fat tails), standardized to unit variance.

    Legacy draws ``rng.standard_t(df, size=n) * sqrt((df-2)/df)`` itself —
    the shared gauss closure is never called (no antithetic/Sobol), so
    ``gauss_mode = "none"`` and evolve owns the draw.
    """

    gauss_mode = "none"
    factors = 1

    def __init__(self, mu: float, sigma: float, dt: float, df: float):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.df = float(df)
        self._scale = math.sqrt((self.df - 2.0) / self.df)
        self._drift_dt = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        self._vol_sqdt = self.sigma * self.sqrt_dt

    def drift(self, state, n):
        return self.mu - 0.5 * self.sigma ** 2

    def diffusion(self, state, n):
        return self.sigma

    def evolve(self, rng, state, z, n):
        shock = rng.standard_t(self.df, size=n) * self._scale
        return self._drift_dt + self._vol_sqdt * shock


class _BootstrapBase(StochasticProcess):
    """Shared pre-processing for the empirical resampling models.

    Matches legacy exactly: returns are centered on their empirical mean,
    a target daily drift (drift-mode + stress-haircut aware) is added back,
    and the stress volatility multiplier scales the centered draw. There is
    no Gaussian decomposition — ``drift``/``diffusion`` are intentionally
    not implemented.
    """

    gauss_mode = "none"
    factors = 1

    def __init__(self, returns: np.ndarray, target_daily_drift: float,
                 vol_multiplier: float = 1.0, dt: float = 1.0 / 252.0):
        super().__init__(dt)
        r = np.asarray(returns, dtype=np.float64).ravel()
        if r.size < 2:
            raise ValueError("bootstrap needs >= 2 historical returns")
        self.emp_mean = float(np.mean(r))
        self.centered = r - self.emp_mean
        self.n_hist = self.centered.size
        self.target_daily = float(target_daily_drift)
        self.vol_mult = float(vol_multiplier)


class HistoricalBootstrapProcess(_BootstrapBase):
    """IID resampling of centered historical daily returns."""

    def evolve(self, rng, state, z, n):
        idx = rng.integers(0, self.n_hist, size=n)
        return self.centered[idx] * self.vol_mult + self.target_daily


class BlockBootstrapProcess(_BootstrapBase):
    """Block resampling: contiguous runs of ``block_length`` days preserve
    short-range autocorrelation. Chunk-local cursors; legacy draws the
    length-``n`` start-index vector only on steps where at least one path
    begins a new block — replicated exactly (including the in-place cursor
    updates)."""

    def __init__(self, returns, target_daily_drift, vol_multiplier=1.0,
                 dt=1.0 / 252.0, *, block_length: int = 20):
        super().__init__(returns, target_daily_drift, vol_multiplier, dt)
        if int(block_length) < 1:
            raise ValueError("block_length must be >= 1")
        self.block_len = int(block_length)

    def init_state(self, n: int) -> Dict[str, np.ndarray]:
        return {
            "cur": np.zeros(n, dtype=np.int64),
            "rem": np.zeros(n, dtype=np.int64),
        }

    def evolve(self, rng, state, z, n):
        cur = state["cur"]
        rem = state["rem"]
        new_mask = rem <= 0
        if new_mask.any():
            starts = rng.integers(0, self.n_hist, size=n)
            cur[new_mask] = starts[new_mask]
            rem[new_mask] = self.block_len
        read = self.centered[cur % self.n_hist]
        cur += 1
        rem -= 1
        return read * self.vol_mult + self.target_daily


class MertonJumpProcess(StochasticProcess):
    """Merton jump-diffusion: Gaussian diffusion + compound-Poisson normal
    jumps, drift compensated so jumps do not add expected return.

    Legacy RNG order per step: diffusion ``standard_normal(n)``, then
    ``poisson(n)`` jump counts, then one ``standard_normal(n)`` used via the
    exact sum-of-normals identity N(k·jm, k·jv²). None of it goes through
    the shared gauss closure.
    """

    gauss_mode = "none"
    factors = 3

    def __init__(self, mu: float, sigma: float, dt: float, *,
                 intensity: float, jump_mean: float, jump_vol: float):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.intensity = float(intensity)
        self.jm = float(jump_mean)
        self.jv = float(jump_vol)
        self._lam_dt = self.intensity * self.dt
        k = math.exp(self.jm + 0.5 * self.jv ** 2) - 1.0
        self._comp_drift = (self.mu - 0.5 * self.sigma ** 2
                            - self.intensity * k) * self.dt
        self._vol_sqdt = self.sigma * self.sqrt_dt

    def drift(self, state, n):
        # Compensated: total expected growth stays mu.
        return self._comp_drift / self.dt

    def diffusion(self, state, n):
        return self.sigma

    def evolve(self, rng, state, z, n):
        zd = rng.standard_normal(n)
        n_jumps = rng.poisson(self._lam_dt, size=n)
        jump = (n_jumps * self.jm
                + self.jv * np.sqrt(n_jumps) * rng.standard_normal(n))
        return self._comp_drift + self._vol_sqdt * zd + jump


class KouJumpProcess(StochasticProcess):
    """Kou double-exponential jump-diffusion (asymmetric up/down jumps).

    The diffusion shock is the shared ``gauss(rng, n)`` — antithetic-aware
    but never Sobol (``gauss_mode = "plain"``, matching the legacy call
    without state/step). Then: ``poisson(n)`` counts; if any jumps, a
    ``random(total)`` up/down branch per jump, an ``exponential`` vector for
    the ups, an ``exponential`` vector for the downs, scatter-added per
    path. Branch order and sizes match legacy exactly.
    """

    gauss_mode = "plain"
    factors = 3

    def __init__(self, mu: float, sigma: float, dt: float, *,
                 intensity: float, p_up: float, eta_up: float,
                 eta_down: float):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.intensity = float(intensity)
        self.p_up = float(p_up)
        self.eta_up = float(eta_up)
        self.eta_down = float(eta_down)
        self._lam_dt = self.intensity * self.dt
        k = (self.p_up * self.eta_up / (self.eta_up - 1.0)
             + (1.0 - self.p_up) * self.eta_down / (self.eta_down + 1.0)) - 1.0
        self._comp_drift = (self.mu - 0.5 * self.sigma ** 2
                            - self.intensity * k) * self.dt
        self._vol_sqdt = self.sigma * self.sqrt_dt

    def drift(self, state, n):
        return self._comp_drift / self.dt

    def diffusion(self, state, n):
        return self.sigma

    def evolve(self, rng, state, z, n):
        n_jumps = rng.poisson(self._lam_dt, size=n)
        total = int(n_jumps.sum())
        jump = np.zeros(n, dtype=np.float64)
        if total > 0:
            up_mask = rng.random(total) < self.p_up
            mags = np.empty(total, dtype=np.float64)
            mags[up_mask] = rng.exponential(1.0 / self.eta_up,
                                            size=int(up_mask.sum()))
            down = ~up_mask
            mags[down] = -rng.exponential(1.0 / self.eta_down,
                                          size=int(down.sum()))
            path_idx = np.repeat(np.arange(n), n_jumps)
            np.add.at(jump, path_idx, mags)
        return self._comp_drift + self._vol_sqdt * z + jump


class GARCHProcess(StochasticProcess):
    """GARCH(1,1): conditional variance follows the realized shock.

    The shock is the fully shared gauss (antithetic AND Sobol-eligible,
    ``gauss_mode = "full"``). The variance recursion updates *after* the
    return is formed, from the realized shock — legacy timing preserved:
    clamp, draw, return, then recurse.
    """

    gauss_mode = "full"
    factors = 1

    def __init__(self, mu: float, sigma: float, dt: float, *,
                 alpha: float, beta: float, omega: Optional[float] = None):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._sigma_daily2 = (self.sigma ** 2) * self.dt
        if omega is not None:
            self.omega = float(omega)
        else:
            self.omega = self._sigma_daily2 * (1.0 - self.alpha - self.beta)
        self._mu_step = self.mu * self.dt

    def init_state(self, n: int) -> Dict[str, np.ndarray]:
        return {"var": np.full(n, self._sigma_daily2, dtype=np.float64)}

    def drift(self, state, n):
        # Per-year rate implied by the per-step mean mu·dt − var/2.
        return self.mu - 0.5 * np.maximum(state["var"], 1e-300) / self.dt

    def diffusion(self, state, n):
        return np.sqrt(np.maximum(state["var"], 1e-300) / self.dt)

    def evolve(self, rng, state, z, n):
        var = state["var"]
        var = np.maximum(var, 1e-300)
        vol = np.sqrt(var)
        shock = vol * z
        log_ret = self._mu_step - 0.5 * var + shock
        state["var"] = self.omega + self.alpha * shock ** 2 + self.beta * var
        return log_ret


class RegimeSwitchingProcess(StochasticProcess):
    """Markov regime switching over (mu, sigma) factor sets.

    Legacy order per step: the transition uniform ``rng.random(n)`` comes
    FIRST (the new regime applies to this step's return), then the
    diffusion ``rng.standard_normal(n)`` — never the shared gauss.
    Chunk-local integer regime state, started in regime 0 ("normal").
    """

    gauss_mode = "none"
    factors = 2

    def __init__(self, mu: float, sigma: float, dt: float, *,
                 mu_factors, sigma_factors, transition):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        mu_f = np.asarray(mu_factors, dtype=np.float64)
        sig_f = np.asarray(sigma_factors, dtype=np.float64)
        self._cum_p = np.cumsum(np.asarray(transition, dtype=np.float64),
                                axis=1)
        self.n_regimes = self._cum_p.shape[0]
        regime_mu = self.mu * mu_f
        regime_sig = self.sigma * sig_f
        self._regime_drift = (regime_mu - 0.5 * regime_sig ** 2) * self.dt
        self._regime_vol = regime_sig * self.sqrt_dt

    def init_state(self, n: int) -> Dict[str, np.ndarray]:
        return {"regime": np.zeros(n, dtype=np.int64)}  # start in "normal"

    def drift(self, state, n):
        return self._regime_drift[state["regime"]] / self.dt

    def diffusion(self, state, n):
        return self._regime_vol[state["regime"]] / self.sqrt_dt

    def evolve(self, rng, state, z, n):
        reg = state["regime"]
        u = rng.random(n)
        new_reg = (u[:, None] >= self._cum_p[reg]).sum(axis=1)
        np.clip(new_reg, 0, self.n_regimes - 1, out=new_reg)
        state["regime"] = new_reg
        zd = rng.standard_normal(n)
        return self._regime_drift[new_reg] + self._regime_vol[new_reg] * zd


# ---------------------------------------------------------------------------
# Config factory (called by mc_engine.process_from_config)
# ---------------------------------------------------------------------------


def extended_process_from_config(cfg: SimulationConfig,
                                 ) -> Optional[StochasticProcess]:
    """Build a Phase-4 process from a legacy SimulationConfig, or None for
    models this module does not know (caller falls back to legacy)."""
    mu = cfg.effective_mu()
    sigma = cfg.effective_sigma()
    model = cfg.model

    if model == MODEL_STUDENT_T:
        return StudentTProcess(mu, sigma, cfg.dt, float(cfg.t_df))

    if model in (MODEL_HIST_BOOTSTRAP, MODEL_BLOCK_BOOTSTRAP):
        r = np.asarray(cfg.historical_returns, dtype=np.float64).ravel()
        emp_mean = float(np.mean(r))
        vol_mult = cfg.stress_vol_multiplier if cfg.stress_enabled else 1.0
        target_daily = _bootstrap_daily_drift(cfg, emp_mean)
        if model == MODEL_HIST_BOOTSTRAP:
            return HistoricalBootstrapProcess(r, target_daily, vol_mult,
                                              cfg.dt)
        return BlockBootstrapProcess(r, target_daily, vol_mult, cfg.dt,
                                     block_length=int(cfg.block_length))

    if model == MODEL_MERTON:
        return MertonJumpProcess(mu, sigma, cfg.dt,
                                 intensity=float(cfg.jump_intensity),
                                 jump_mean=float(cfg.jump_mean),
                                 jump_vol=float(cfg.jump_vol))

    if model == MODEL_KOU:
        return KouJumpProcess(mu, sigma, cfg.dt,
                              intensity=float(cfg.kou_intensity),
                              p_up=float(cfg.kou_p_up),
                              eta_up=float(cfg.kou_eta_up),
                              eta_down=float(cfg.kou_eta_down))

    if model == MODEL_GARCH:
        return GARCHProcess(mu, sigma, cfg.dt,
                            alpha=float(cfg.garch_alpha),
                            beta=float(cfg.garch_beta),
                            omega=(float(cfg.garch_omega)
                                   if cfg.garch_omega is not None else None))

    if model == MODEL_REGIME:
        preset = REGIME_PRESETS[cfg.regime_preset]
        return RegimeSwitchingProcess(
            mu, sigma, cfg.dt,
            mu_factors=preset["mu_factors"],
            sigma_factors=preset["sigma_factors"],
            transition=preset["transition"],
        )

    return None
