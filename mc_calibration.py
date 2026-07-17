"""
Simple model calibration helpers (Phase 2).

Provides lightweight MLE / moment estimators for:
  * GARCH(1,1) on daily log-returns
  * Heston parameters (method-of-moments style on realized variance)

Design goals
------------
* Optional SciPy (``scipy.optimize``) for proper MLE; pure-NumPy fallbacks
  when SciPy is missing so imports and CI stay green offline.
* Never mutates ``mc_core`` defaults — callers pass results into
  ``SimulationConfig`` themselves.
* Laptop-safe: operates on 1-D return series only (no path matrices).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class GarchParams:
    """GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2."""

    omega: float
    alpha: float
    beta: float
    loglik: float
    method: str  # "mle" | "variance_target" | "fallback"
    n_obs: int
    note: str = ""

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    def as_config_kwargs(self) -> Dict[str, float]:
        return {
            "garch_omega": float(self.omega),
            "garch_alpha": float(self.alpha),
            "garch_beta": float(self.beta),
        }


@dataclass
class HestonParams:
    """Heston: dS = mu S dt + sqrt(v) S dW1, dv = kappa(theta-v)dt + xi sqrt(v) dW2."""

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    method: str
    n_obs: int
    note: str = ""

    def as_config_kwargs(self) -> Dict[str, float]:
        return {
            "heston_kappa": float(self.kappa),
            "heston_theta": float(self.theta),
            "heston_xi": float(self.xi),
            "heston_rho": float(self.rho),
            "heston_v0": float(self.v0),
        }


def _scipy_minimize_available() -> bool:
    try:  # pragma: no cover - env dependent
        from scipy.optimize import minimize  # noqa: F401
        return True
    except Exception:
        return False


def _garch_loglik(
    params: Tuple[float, float, float],
    returns: np.ndarray,
) -> float:
    """Negative average log-likelihood of Gaussian GARCH(1,1) (for minimizers)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e12
    r = returns
    n = r.size
    # unconditional variance start
    var = float(np.var(r)) if n > 1 else omega / max(1e-8, 1.0 - alpha - beta)
    var = max(var, 1e-12)
    ll = 0.0
    for t in range(n):
        # Gaussian log density of r_t | F_{t-1}
        ll += -0.5 * (math.log(2.0 * math.pi) + math.log(var) + (r[t] ** 2) / var)
        var = omega + alpha * (r[t] ** 2) + beta * var
        var = max(var, 1e-12)
    return -ll / n  # minimize average NLL


def calibrate_garch(
    returns: np.ndarray,
    *,
    demean: bool = True,
) -> GarchParams:
    """
    Calibrate GARCH(1,1) to a 1-D daily log-return series.

    Tries SciPy MLE first; falls back to a variance-targeting heuristic
    (alpha=0.08, beta=0.90 style, omega chosen so long-run var = sample var).
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if demean and r.size:
        r = r - float(np.mean(r))
    n = int(r.size)
    if n < 30:
        sample_var = float(np.var(r)) if n > 1 else 1e-4
        alpha, beta = 0.08, 0.90
        omega = sample_var * (1.0 - alpha - beta)
        return GarchParams(
            omega=max(omega, 1e-12), alpha=alpha, beta=beta,
            loglik=float("nan"), method="fallback", n_obs=n,
            note="Too few observations for MLE; used default persistence.",
        )

    sample_var = float(np.var(r, ddof=1))
    # Heuristic seed
    alpha0, beta0 = 0.08, 0.90
    omega0 = max(sample_var * (1.0 - alpha0 - beta0), 1e-12)

    if _scipy_minimize_available():
        from scipy.optimize import minimize  # type: ignore

        x0 = np.array([omega0, alpha0, beta0], dtype=float)
        bounds = [(1e-14, None), (1e-8, 0.5), (1e-8, 0.99)]

        def cons_persist(x):
            return 0.999 - x[1] - x[2]

        try:
            res = minimize(
                _garch_loglik, x0, args=(r,),
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": cons_persist},
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if res.success:
                omega, alpha, beta = [float(v) for v in res.x]
                nll = float(res.fun)
                return GarchParams(
                    omega=omega, alpha=alpha, beta=beta,
                    loglik=-nll * n, method="mle", n_obs=n,
                )
        except Exception as exc:  # noqa: BLE001
            note = f"MLE failed ({type(exc).__name__}); variance-target fallback."
        else:
            note = "MLE did not converge; variance-target fallback."
    else:
        note = "SciPy unavailable; variance-target fallback."

    alpha, beta = alpha0, beta0
    omega = max(sample_var * (1.0 - alpha - beta), 1e-12)
    nll = _garch_loglik((omega, alpha, beta), r)
    return GarchParams(
        omega=omega, alpha=alpha, beta=beta,
        loglik=-nll * n, method="variance_target", n_obs=n, note=note,
    )


def calibrate_heston(
    returns: np.ndarray,
    *,
    dt: float = 1.0 / 252.0,
) -> HestonParams:
    """
    Rough Heston calibration from daily returns via realized-variance moments.

    This is intentionally simple (not a full options-surface calibration):
      * theta, v0 ≈ sample variance of returns / dt  (annualized variance level)
      * kappa from AR(1)-like persistence of a rolling realized variance
      * xi from volatility-of-volatility of that series
      * rho from corr(return, delta realized variance)

    Good enough to seed ``SimulationConfig``; not a production vol-surface fit.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = int(r.size)
    # Annualized variance level
    daily_var = float(np.var(r, ddof=1)) if n > 1 else 1e-4
    theta = max(daily_var / max(dt, 1e-12), 1e-8)
    v0 = theta

    # Rolling realized variance (21-day) as a proxy for spot variance path
    win = min(21, max(5, n // 10))
    if n < win + 5:
        return HestonParams(
            kappa=1.5, theta=theta, xi=0.3, rho=-0.5, v0=v0,
            method="fallback", n_obs=n,
            note="Too few observations; used default Heston shape params.",
        )

    rv = np.array(
        [float(np.var(r[i - win:i])) / dt for i in range(win, n)],
        dtype=float,
    )
    rv = np.maximum(rv, 1e-10)
    # AR(1) persistence on rv: rv_t = c + phi rv_{t-1}
    x = rv[:-1]
    y = rv[1:]
    if x.size > 2 and float(np.var(x)) > 0:
        phi = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        phi = min(max(phi, 0.0), 0.99)
        # phi ≈ exp(-kappa * dt_rv); dt_rv ≈ 1 trading day
        kappa = max(-math.log(max(phi, 1e-6)) / max(dt, 1e-12) * dt, 0.05)
        # more direct: kappa ≈ (1-phi)/dt_day with dt_day=1/252 already in theta units
        kappa = max((1.0 - phi) / 1.0, 0.05)  # per day in "rv step" units → clamp later
        # Convert daily mean-reversion speed to annual: multiply by 252 roughly
        kappa = float(min(max(kappa * 252.0 * dt / dt, 0.1), 10.0))
        # Simpler stable mapping used in practice for daily steps:
        kappa = float(min(max(-math.log(max(phi, 1e-6)) * 252.0, 0.1), 10.0))
    else:
        kappa = 1.5
        phi = 0.9

    # vol-of-vol from residual std of rv changes
    drv = np.diff(rv)
    if drv.size > 2:
        # Euler: dv ≈ ... + xi sqrt(v) dW; so xi ≈ std(dv) / mean(sqrt(v)) / sqrt(dt_step)
        xi = float(np.std(drv, ddof=1) / (float(np.mean(np.sqrt(rv[:-1]))) + 1e-12))
        xi = float(min(max(xi, 0.05), 2.0))
    else:
        xi = 0.3

    # leverage correlation
    r_aligned = r[win:win + drv.size]
    if r_aligned.size == drv.size and r_aligned.size > 5:
        c = np.corrcoef(r_aligned, drv)[0, 1]
        rho = float(c) if np.isfinite(c) else -0.5
        rho = float(min(max(rho, -0.95), 0.95))
    else:
        rho = -0.5

    return HestonParams(
        kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0,
        method="moments", n_obs=n,
        note="Moment-based Heston seed from rolling realized variance.",
    )


def calibrate_from_prices(
    prices: np.ndarray,
    *,
    model: str = "garch",
    dt: float = 1.0 / 252.0,
) -> Dict[str, Any]:
    """Convenience: prices → log returns → calibrate ``garch`` or ``heston``."""
    p = np.asarray(prices, dtype=float).ravel()
    if p.size < 3 or np.any(p <= 0):
        raise ValueError("need at least 3 strictly positive prices")
    rets = np.diff(np.log(p))
    model_l = model.strip().lower()
    if model_l in ("garch", "garch(1,1)", "garch11"):
        g = calibrate_garch(rets)
        return {"model": "garch", "params": g, "kwargs": g.as_config_kwargs()}
    if model_l in ("heston",):
        h = calibrate_heston(rets, dt=dt)
        return {"model": "heston", "params": h, "kwargs": h.as_config_kwargs()}
    raise ValueError(f"Unknown calibration model: {model!r}")
