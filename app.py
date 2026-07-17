"""Streamlit GUI for the memory-safe Monte Carlo GBM simulator.

Launch with::

    streamlit run app.py

All UI logic lives inside :func:`main`, which only runs when the script is
executed by Streamlit (``__name__ == "__main__"``).  This keeps ``import app``
cheap and side-effect free so it can be imported in tests and other tools.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

import mc_core


# ---------------------------------------------------------------------------
# Pure helpers (import-safe, unit-testable without a running Streamlit server)
# ---------------------------------------------------------------------------


def build_config_from_inputs(
    *,
    ticker: str,
    s0: float,
    paths: int,
    horizon: int,
    mu: float,
    sigma: float,
    chunk_size: int,
    seed: Optional[int],
    cost: float,
    sample_paths: int = 50,
    **model_kwargs,
) -> mc_core.SimulationConfig:
    """Assemble a validated :class:`mc_core.SimulationConfig` from GUI inputs.

    Extra ``model_kwargs`` (model, drift_mode, t_df, historical_returns,
    block_length, jump_*, regime_preset, stress_*) are passed straight through to
    :class:`mc_core.SimulationConfig`.
    """

    return mc_core.SimulationConfig(
        ticker=ticker or "ASSET",
        s0=float(s0),
        paths=int(paths),
        horizon=int(horizon),
        mu=float(mu),
        sigma=float(sigma),
        chunk_size=int(chunk_size),
        seed=seed,
        cost=float(cost),
        sample_paths=int(sample_paths),
        **model_kwargs,
    ).validate()


def percentile_table_rows(result: mc_core.SimulationResult):
    """Return percentile-table rows as ``(label, value)`` tuples."""
    return [(f"p{p}", v) for p, v in result.stats["percentiles"].items()]


def path_mode_settings(mode: str):
    """Return ``(min_paths, max_paths, default_paths, step)`` for a path mode.

    * Preview / Standard / Serious -> editable in [1,000; 1,000,000], prefilled
      with the preset value (10,000 / 100,000 / 1,000,000).
    * Custom -> editable in [1,000; 1,000,000], prefilled with 250,000.
    * Tail-risk (advanced) -> editable in [2,000,000; 5,000,000], prefilled with
      2,000,000.
    """

    if mode not in mc_core.PATH_MODES:
        raise ValueError(f"Unknown path mode: {mode!r}")
    if mode == "Tail-risk (advanced)":
        return (
            mc_core.TAIL_RISK_MIN_PATHS,
            mc_core.TAIL_RISK_MAX_PATHS,
            mc_core.TAIL_RISK_MIN_PATHS,
            500_000,
        )
    if mode == "Custom":
        return (mc_core.CUSTOM_MIN_PATHS, mc_core.CUSTOM_MAX_PATHS, 250_000, 5_000)
    # Preview / Standard / Serious presets
    return (
        mc_core.CUSTOM_MIN_PATHS,
        mc_core.CUSTOM_MAX_PATHS,
        int(mc_core.PATH_MODES[mode]),
        1_000,
    )


def resolve_path_count(mode: str, explicit_paths=None) -> int:
    """Resolve and validate the effective path count for the GUI.

    This is the small, Streamlit-free helper that the GUI uses and that tests can
    call directly.  When ``explicit_paths`` is ``None`` the mode's default/preset
    is used (so Preview/Standard/Serious resolve to 10,000/100,000/1,000,000).
    When ``explicit_paths`` is supplied it is validated against the mode's safe
    range, mirroring what the editable "Number of paths" field allows.
    """

    min_paths, max_paths, default_paths, _ = path_mode_settings(mode)
    paths = int(default_paths if explicit_paths is None else explicit_paths)
    if not (min_paths <= paths <= max_paths):
        raise ValueError(
            f"{mode} mode requires paths between {min_paths:,} and {max_paths:,} "
            f"(got {paths:,})."
        )
    return paths


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - exercised via `streamlit run`
    import streamlit as st
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Monte Carlo Market-Risk Simulator", layout="wide")
    st.title("Monte Carlo Market-Risk Simulator")
    st.caption(
        "CPU-first, memory-safe multi-model market-risk simulator: GBM, Student-t, "
        "Historical/Block Bootstrap, Merton Jump-Diffusion, and Regime Switching. "
        "Never allocates a full path x step matrix."
    )

    # ---------------------- Sidebar inputs ----------------------
    with st.sidebar:
        st.header("Inputs")

        # ---------------------- Model selection (top of sidebar) ----------------------
        st.subheader("Model")
        model = st.selectbox(
            "Simulation model", list(mc_core.MODELS), index=0,
            help="Choose the stochastic model used to generate price paths.",
        )

        model_inputs: dict = {"model": model}
        if model == mc_core.MODEL_STUDENT_T:
            model_inputs["t_df"] = st.number_input(
                "Student-t degrees of freedom", min_value=2.5, max_value=100.0,
                value=5.0, step=0.5,
                help="Lower = fatter tails. Shocks are standardized to unit variance.",
            )
        elif model == mc_core.MODEL_BLOCK_BOOTSTRAP:
            model_inputs["block_length"] = st.number_input(
                "Block length (days)", min_value=1, max_value=120, value=20, step=1,
                help="Preserves short-term volatility clustering.",
            )
        elif model == mc_core.MODEL_MERTON:
            jump_preset = st.selectbox("Jump preset", ["stock", "crypto", "custom"], index=0)
            base = mc_core.JUMP_PRESETS.get(
                jump_preset, mc_core.JUMP_PRESETS["stock"]
            )
            disabled = jump_preset != "custom"
            model_inputs["jump_intensity"] = st.number_input(
                "Jump intensity (per year)", min_value=0.0, max_value=100.0,
                value=float(base["intensity"]), step=0.5, disabled=disabled,
            )
            model_inputs["jump_mean"] = st.number_input(
                "Jump mean (log)", min_value=-1.0, max_value=1.0,
                value=float(base["mean"]), step=0.01, format="%.3f", disabled=disabled,
            )
            model_inputs["jump_vol"] = st.number_input(
                "Jump volatility (log)", min_value=0.0, max_value=1.0,
                value=float(base["vol"]), step=0.01, format="%.3f", disabled=disabled,
            )
        elif model == mc_core.MODEL_REGIME:
            model_inputs["regime_preset"] = st.selectbox(
                "Regime preset", list(mc_core.REGIME_PRESETS), index=0,
                help="Crypto preset spends more time in high-vol/crash regimes.",
            )
        elif model == mc_core.MODEL_HESTON:
            model_inputs["heston_kappa"] = st.number_input(
                "Heston kappa (mean reversion)", min_value=0.0, max_value=20.0,
                value=1.5, step=0.1,
            )
            model_inputs["heston_theta"] = st.number_input(
                "Heston theta (long-run var, 0 = use sigma^2)",
                min_value=0.0, max_value=2.0, value=0.0, step=0.01, format="%.4f",
            ) or None
            model_inputs["heston_xi"] = st.number_input(
                "Heston xi (vol-of-vol)", min_value=0.0, max_value=5.0,
                value=0.3, step=0.05,
            )
            model_inputs["heston_rho"] = st.number_input(
                "Heston rho (corr S,v)", min_value=-1.0, max_value=1.0,
                value=-0.7, step=0.05,
            )
            model_inputs["heston_v0"] = st.number_input(
                "Heston v0 (initial var, 0 = use sigma^2)",
                min_value=0.0, max_value=2.0, value=0.0, step=0.01, format="%.4f",
            ) or None
        elif model == mc_core.MODEL_GARCH:
            model_inputs["garch_alpha"] = st.number_input(
                "GARCH alpha", min_value=0.0, max_value=0.99, value=0.08, step=0.01,
            )
            model_inputs["garch_beta"] = st.number_input(
                "GARCH beta", min_value=0.0, max_value=0.99, value=0.90, step=0.01,
            )
            st.caption("Requires alpha + beta < 1 for stationarity.")
        elif model == mc_core.MODEL_KOU:
            kou_preset = st.selectbox("Kou preset", ["stock", "crypto", "custom"], index=0)
            base = {"stock": dict(intensity=1.0, p_up=0.4, eta_up=25.0, eta_down=15.0),
                    "crypto": dict(intensity=6.0, p_up=0.45, eta_up=12.0, eta_down=8.0)}.get(
                        kou_preset, {})
            disabled = kou_preset != "custom"
            model_inputs["kou_intensity"] = st.number_input(
                "Kou jump intensity (per year)", min_value=0.0, max_value=100.0,
                value=float(base.get("intensity", 1.0)), step=0.5, disabled=disabled,
            )
            model_inputs["kou_p_up"] = st.number_input(
                "Kou P(jump up)", min_value=0.0, max_value=1.0,
                value=float(base.get("p_up", 0.4)), step=0.05, disabled=disabled,
            )
            model_inputs["kou_eta_up"] = st.number_input(
                "Kou eta_up (up-jump rate)", min_value=1.1, max_value=100.0,
                value=float(base.get("eta_up", 25.0)), step=1.0, disabled=disabled,
            )
            model_inputs["kou_eta_down"] = st.number_input(
                "Kou eta_down (down-jump rate)", min_value=0.1, max_value=100.0,
                value=float(base.get("eta_down", 15.0)), step=1.0, disabled=disabled,
            )
        elif model in mc_core.BOOTSTRAP_MODELS:
            st.caption("Uses empirical daily returns sampled from history.")

        # ---------------------- Conservative drift mode ----------------------
        st.subheader("Drift mode")
        drift_mode = st.selectbox(
            "Conservative drift mode", list(mc_core.DRIFT_MODES), index=0,
            help="Reduce reliance on historical drift for more conservative outlooks.",
        )
        manual_drift = None
        if drift_mode == mc_core.DRIFT_MANUAL:
            manual_drift = st.number_input(
                "Manual annual drift", min_value=-1.0, max_value=2.0,
                value=0.0, step=0.01, format="%.3f",
            )

        st.subheader("Asset")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        years = st.number_input(
            "Years of history", min_value=1.0, max_value=30.0, value=3.0, step=1.0
        )
        s0_override = st.number_input(
            "Starting price override (0 = use market)",
            min_value=0.0, value=0.0, step=1.0,
        )

        st.subheader("Paths")
        mode = st.selectbox(
            "Path mode", list(mc_core.PATH_MODES.keys()), index=1, key="path_mode",
            help=(
                "Preview/Standard/Serious prefill a preset you can still edit. "
                "Custom: 1,000-1,000,000. Tail-risk advanced: 2,000,000-5,000,000."
            ),
        )

        # Per-mode editable range and the value to prefill when the mode changes.
        min_paths, max_paths, default_paths, path_step = path_mode_settings(mode)

        # Snap the editable field to the mode's default whenever the mode changes,
        # but otherwise preserve whatever the user typed.
        if st.session_state.get("_last_path_mode") != mode:
            st.session_state["num_paths"] = int(default_paths)
            st.session_state["_last_path_mode"] = mode

        edited_paths = st.number_input(
            "Number of paths",
            min_value=int(min_paths),
            max_value=int(max_paths),
            step=int(path_step),
            key="num_paths",
            help="Type any value within the selected mode's safe range.",
        )
        # Validate through the same Streamlit-free helper the tests exercise.
        try:
            paths = resolve_path_count(mode, edited_paths)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.caption(f"Selected paths: {paths:,}")

        if mode == "Tail-risk (advanced)":
            st.warning(
                "TAIL-RISK ADVANCED MODE: this allocates very large simulations "
                "(2,000,000-5,000,000 paths). It is CPU/RAM intensive and slower. "
                "Chunked execution keeps memory bounded, but expect a longer runtime."
            )
        else:
            warn = mc_core.tail_risk_warning(paths)
            if warn:
                st.warning(warn)

        st.subheader("Simulation")
        horizon = st.number_input(
            "Horizon (trading-day steps)", min_value=1, max_value=2520,
            value=252, step=1,
        )
        default_chunk = min(mc_core.DEFAULT_SERIOUS_CHUNK_SIZE, int(paths))
        chunk_size = st.number_input(
            "Chunk size (paths per chunk)",
            min_value=1_000, max_value=200_000, value=default_chunk, step=5_000,
            help="Serious mode default is 25,000-50,000 to bound peak memory.",
        )
        seed_text = st.text_input("Seed (blank = random)", value="42")
        cost = st.number_input(
            "Transaction cost / slippage (fraction)",
            min_value=0.0, max_value=0.5, value=0.0, step=0.0005, format="%.4f",
        )

        # ---------------------- Stress overlay ----------------------
        st.subheader("Stress overlay")
        stress_enabled = st.checkbox("Enable deterministic stress overlay", value=False)
        stress_crash = stress_vol_mult = stress_haircut = 0.0
        stress_vol_mult = 1.0
        if stress_enabled:
            stress_crash = st.number_input(
                "One-day crash on day 1 (fraction)", min_value=0.0, max_value=0.95,
                value=0.0, step=0.05,
            )
            stress_vol_mult = st.number_input(
                "Volatility multiplier", min_value=0.1, max_value=10.0,
                value=1.0, step=0.1,
            )
            stress_haircut = st.number_input(
                "Drift haircut (fraction removed)", min_value=0.0, max_value=1.0,
                value=0.0, step=0.05,
            )

        # ---------------------- Advanced Math panel ----------------------
        st.subheader("Advanced math")
        variance_reduction = st.selectbox(
            "Variance reduction", list(mc_core.VARIANCE_REDUCTION_METHODS), index=0,
            help="Antithetic and control-variate run on CPU; Sobol needs scipy.",
        )
        if variance_reduction == mc_core.VR_SOBOL and not mc_core.sobol_available():
            st.caption("Sobol unavailable (scipy not installed); will fall back.")
        evt_enabled = st.checkbox("EVT tail analysis", value=False,
                                  help="Fit a Generalized Pareto tail to simulated losses.")
        portfolio_enabled = st.checkbox("Portfolio mode (multi-asset)", value=False)
        ruin_threshold = st.number_input(
            "Risk-of-ruin threshold (fraction of S0)",
            min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        )

        with st.expander("Advanced parameter overrides"):
            mu_override = st.text_input("mu override (annual, blank = estimate)", value="")
            sigma_override = st.text_input("sigma override (annual, blank = estimate)", value="")

        run_clicked = st.button("Run simulation", type="primary")

    seed = int(seed_text) if seed_text.strip() else None

    # Conservative-drift warning surfaced near the top of the main panel.
    _eff_mu_preview = mc_core.SimulationConfig(
        mu=(float(mu_override) if mu_override.strip() else mc_core.FALLBACK_MU),
        drift_mode=drift_mode, manual_drift=manual_drift,
    ).effective_mu()
    if _eff_mu_preview > mc_core.HIGH_DRIFT_WARNING_LEVEL:
        st.warning(
            f"Conservative-drift warning: effective annual drift ~{_eff_mu_preview:.0%} "
            "is very high. Consider Half/Zero drift for a less optimistic view."
        )

    # ---------------------- Memory pre-flight ----------------------
    preview_cfg = mc_core.SimulationConfig(
        paths=int(paths), horizon=int(horizon), chunk_size=int(chunk_size)
    )
    preview_mem = mc_core.predict_memory(preview_cfg)
    st.info(preview_mem.status())

    # ------------------------------------------------------------------
    # Two modes: single-model analysis and side-by-side model comparison.
    # ------------------------------------------------------------------
    shared = dict(
        ticker=ticker, years=years, s0_override=s0_override,
        paths=int(paths), horizon=int(horizon), chunk_size=int(chunk_size),
        seed=seed, cost=cost, drift_mode=drift_mode, manual_drift=manual_drift,
        mu_override=mu_override, sigma_override=sigma_override,
        stress_enabled=stress_enabled, stress_crash=stress_crash,
        stress_vol_mult=stress_vol_mult, stress_haircut=stress_haircut,
        variance_reduction=variance_reduction, ruin_threshold=ruin_threshold,
        evt_enabled=evt_enabled,
    )

    tab_names = ["Single model", "Model comparison", "Investment Report", "Tactical"]
    if portfolio_enabled:
        tab_names.append("Portfolio")
    _tabs = st.tabs(tab_names)
    tab_single, tab_compare, tab_report, tab_tactical = (
        _tabs[0], _tabs[1], _tabs[2], _tabs[3]
    )
    tab_portfolio = _tabs[4] if portfolio_enabled else None

    with tab_single:
        # Run only when the button is clicked; persist the result in
        # session_state so later reruns (e.g. clicking a download button) keep
        # showing the results instead of resetting to the initial screen.
        if run_clicked:
            with st.spinner(f"Fetching parameters for {ticker}..."):
                market = mc_core.estimate_parameters_from_history(
                    ticker, years=years,
                    s0_override=(s0_override if s0_override > 0 else None),
                )
            mu = float(mu_override) if mu_override.strip() else market.mu
            sigma = float(sigma_override) if sigma_override.strip() else market.sigma
            try:
                config = build_config_from_inputs(
                    ticker=ticker, s0=market.s0, paths=int(paths), horizon=int(horizon),
                    mu=mu, sigma=sigma, chunk_size=int(chunk_size), seed=seed, cost=cost,
                    drift_mode=drift_mode, manual_drift=manual_drift,
                    historical_returns=market.daily_log_returns,
                    stress_enabled=stress_enabled,
                    stress_crash_pct=stress_crash,
                    stress_vol_multiplier=stress_vol_mult,
                    stress_drift_haircut=stress_haircut,
                    variance_reduction=variance_reduction,
                    ruin_threshold=ruin_threshold,
                    **model_inputs,
                )
            except ValueError as exc:
                st.error(f"Invalid configuration: {exc}")
                st.stop()
            with st.spinner(f"Running {config.model} on {paths:,} paths "
                            f"in chunks of {chunk_size:,}..."):
                result = mc_core.simulate(config)
            st.session_state["mc_result"] = result
            st.session_state["mc_market"] = market
            st.session_state["mc_evt"] = (
                mc_core.evt_from_result(result) if evt_enabled else None
            )

        result = st.session_state.get("mc_result")
        market = st.session_state.get("mc_market")
        if result is None:
            st.write("Configure inputs in the sidebar and click **Run simulation**.")
        else:
            _render_single_model(st, pd, plt, result, market,
                                 st.session_state.get("mc_evt"))

    with tab_compare:
        _render_comparison(st, pd, shared)

    with tab_report:
        _render_investment_report(st, pd, shared)

    with tab_tactical:
        _render_tactical(st, pd, shared)

    if tab_portfolio is not None:
        with tab_portfolio:
            _render_portfolio(st, pd, shared)


def _render_single_model(st, pd, plt, result, market, evt=None) -> None:
    """Render the full single-model output panel from a stored result."""
    config = result.config
    s = result.stats
    ticker = config.ticker
    paths = config.paths

    # ---------------------- Data source notice ----------------------
    if market is not None:
        if market.source == "fallback":
            st.warning(
                f"Market data unavailable; using fallback parameters. {market.note}"
            )
        else:
            st.success(
                f"Estimated from {market.source}: S0={config.s0:,.2f}, "
                f"mu={config.mu:.2%}, sigma={config.sigma:.2%}"
            )

    # ---------------------- Validation warnings ----------------------
    warnings = mc_core.collect_warnings(config, market, evt=evt)
    for w in warnings:
        st.warning(w)

    # ---------------------- Selected model & assumptions ----------------------
    assumptions = mc_core.model_assumptions(config, market)
    st.subheader("Selected model & assumptions")
    st.info(
        f"**Model:** {assumptions['model']}  |  **Drift mode:** {assumptions['drift_mode']}  "
        f"|  **Volatility source:** {assumptions['volatility_source']}  |  "
        f"**Effective mu/sigma:** {assumptions['effective_mu_annual']:.2%} / "
        f"{assumptions['effective_sigma_annual']:.2%}"
        + (f"  |  **Stress:** crash {assumptions['stress']['one_day_crash_pct']:.0%}, "
           f"vol x{assumptions['stress']['vol_multiplier']:g}, "
           f"drift haircut {assumptions['stress']['drift_haircut']:.0%}"
           if assumptions["stress"]["enabled"] else "")
    )

    # ---------------------- Headline metrics ----------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected ending value", f"{s['expected_value']:,.2f}",
              f"{s['expected_return']:+.2%}")
    c2.metric("Median ending value", f"{s['median_value']:,.2f}")
    c3.metric("Probability of profit", f"{s['prob_profit']:.2%}")
    c4.metric("Probability of loss", f"{s['prob_loss']:.2%}")

    # ---------------------- Probability buckets ----------------------
    st.subheader("Probability buckets")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("P(ending > +20%)", f"{s['prob_gain_20']:.2%}")
    b2.metric("P(ending < -10%)", f"{s['prob_loss_10']:.2%}")
    b3.metric("P(ending < -20%)", f"{s['prob_loss_20']:.2%}")
    b4.metric(f"P({int(s['drawdown_threshold']*100)}% drawdown)",
              f"{s.get('prob_drawdown', float('nan')):.2%}")
    b5.metric("Worst 1% avg value", f"{s['worst_1pct_avg_value']:,.2f}")

    # ---------------------- Risk tables ----------------------
    st.subheader("Risk metrics (loss relative to starting price)")
    risk_rows = []
    for level in mc_core.RISK_LEVELS:
        key = mc_core._level_key(level)
        risk_rows.append({
            "Confidence": f"{key}%",
            "VaR (value)": s["var"][key]["value"],
            "VaR (%)": s["var"][key]["pct"],
            "Expected Shortfall (value)": s["expected_shortfall"][key]["value"],
            "Expected Shortfall (%)": s["expected_shortfall"][key]["pct"],
        })
    st.dataframe(pd.DataFrame(risk_rows), hide_index=True, use_container_width=True)

    st.subheader("Percentile table (ending value)")
    pct_df = pd.DataFrame(percentile_table_rows(result), columns=["Percentile", "Value"])
    st.dataframe(pct_df, hide_index=True, use_container_width=True)

    # ---------------------- Advanced risk metrics ----------------------
    st.subheader("Advanced risk metrics")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Sharpe (annual)", f"{s.get('sharpe_annual', 0):.2f}")
    a2.metric("Sortino (annual)", f"{s.get('sortino_annual', 0):.2f}")
    a3.metric("Calmar", f"{s.get('calmar', 0):.2f}")
    a4.metric("Mean max drawdown", f"{s.get('mean_max_drawdown', 0):.2%}")
    a5, a6, a7, a8 = st.columns(4)
    a5.metric("Mean DD duration (steps)", f"{s.get('mean_drawdown_duration', 0):.1f}")
    a6.metric(f"P(ruin < {int(config.ruin_threshold*100)}% S0)",
              f"{s.get('prob_ruin', 0):.2%}")
    a7.metric("Kelly fraction", f"{s.get('kelly_fraction', 0):.2f}")
    a8.metric("Annualized return", f"{s.get('annualized_return', 0):+.2%}")
    st.caption(f"Note: {mc_core.KELLY_WARNING}")

    # ---------------------- EVT tail risk ----------------------
    if evt is not None:
        st.subheader("EVT tail risk (Generalized Pareto over loss exceedances)")
        if evt.get("error"):
            st.warning(f"EVT could not be computed: {evt['error']}")
        else:
            evt_rows = []
            for key in ("95", "99", "99.5", "99.9"):
                if key in evt["var"]:
                    evt_rows.append({
                        "Confidence": f"{key}%",
                        "EVT tail loss (return)": evt["var"][key],
                        "EVT Expected Shortfall": evt["es"][key],
                    })
            st.dataframe(pd.DataFrame(evt_rows), hide_index=True, use_container_width=True)
            st.caption(
                f"Threshold (loss): {evt.get('threshold', float('nan')):.4f} | "
                f"exceedances: {evt.get('n_exceedances', 0)} | "
                f"GPD shape xi: {evt.get('shape_xi', float('nan')):.3f}"
            )
            if evt.get("warning"):
                st.warning(evt["warning"])

    # ---------------------- Charts ----------------------
    st.subheader("Charts")
    col_a, col_b = st.columns(2)

    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        traj = result.sample_trajectories
        for i in range(traj.shape[0]):
            ax1.plot(traj[i], linewidth=0.7, alpha=0.7)
        ax1.set_title(f"Sample paths ({traj.shape[0]} of {paths:,})")
        ax1.set_xlabel("step")
        ax1.set_ylabel("price")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(result.final_values, bins=80, color="#3b7dd8", alpha=0.85)
        ax2.axvline(config.s0, color="black", linestyle="--", linewidth=1, label="S0")
        ax2.set_title("Ending-value distribution")
        ax2.set_xlabel("ending value")
        ax2.set_ylabel("frequency")
        ax2.legend()
        st.pyplot(fig2)
        plt.close(fig2)

    if result.convergence_paths.size > 1:
        st.subheader("Convergence of expected ending value")
        conv_df = pd.DataFrame(
            {"mean_ending_value": result.convergence_means},
            index=pd.Index(result.convergence_paths, name="paths"),
        )
        st.line_chart(conv_df)

    # ---------------------- Runtime + memory ----------------------
    st.subheader("Runtime & memory safety")
    rc1, rc2 = st.columns(2)
    rc1.metric("Runtime", f"{result.runtime_seconds:.3f} s")
    rc2.metric("Throughput", f"{s.get('paths_per_second', 0):,.0f} paths/s")
    if result.memory.is_chunk_safe:
        st.success(result.memory.status())
    else:
        st.error(result.memory.status())

    # ---------------------- Exports ----------------------
    st.subheader("Export")
    csv_text = mc_core.report_to_csv(result)
    json_text = mc_core.report_to_json(result, market, evt=evt)
    e1, e2 = st.columns(2)
    e1.download_button(
        "Download CSV summary", data=csv_text,
        file_name=f"{ticker}_mc_summary.csv", mime="text/csv",
    )
    e2.download_button(
        "Download JSON report", data=json_text,
        file_name=f"{ticker}_mc_report.json", mime="application/json",
    )


COMPARISON_DISPLAY_NAMES = {
    "model": "Model",
    "expected_ending_value": "Expected",
    "median_ending_value": "Median",
    "prob_profit": "P(profit)",
    "prob_loss": "P(loss)",
    "prob_gain_20": "P(gain>20%)",
    "prob_loss_10": "P(loss>10%)",
    "prob_loss_20": "P(loss>20%)",
    "prob_drawdown_50": "P(50% DD)",
    "percentile_5": "P5",
    "percentile_95": "P95",
    "var_99": "VaR 99%",
    "es_99": "ES 99%",
    "evt_var_99": "EVT VaR 99%",
    "evt_es_99": "EVT ES 99%",
    "max_drawdown_prob": "Max DD prob",
    "prob_ruin": "P(ruin)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "disagreement_rank": "Disagree rank",
    "runtime_seconds": "Runtime (s)",
    "chunk_safe": "Chunk-safe",
}


def _render_comparison(st, pd, shared: dict) -> None:
    """Render the Model Comparison tab: pick models, run, compare, export."""
    st.subheader("Model comparison")
    st.caption(
        "Run several models on identical ticker/paths/horizon/chunk/seed/drift "
        "settings and compare their risk profiles side-by-side. Each model uses "
        "the same chunk-safe engine (no full path x step matrix)."
    )

    selected = st.multiselect(
        "Models to compare", list(mc_core.MODELS), default=list(mc_core.MODELS),
        help="All available models are selected by default.",
    )
    run_compare = st.button("Run comparison", type="primary", key="run_comparison")

    if run_compare:
        if not selected:
            st.error("Select at least one model to compare.")
            st.stop()
        with st.spinner(f"Fetching parameters for {shared['ticker']}..."):
            market = mc_core.estimate_parameters_from_history(
                shared["ticker"], years=shared["years"],
                s0_override=(shared["s0_override"] if shared["s0_override"] > 0 else None),
            )
        mu = float(shared["mu_override"]) if shared["mu_override"].strip() else market.mu
        sigma = float(shared["sigma_override"]) if shared["sigma_override"].strip() else market.sigma
        try:
            base_config = build_config_from_inputs(
                ticker=shared["ticker"], s0=market.s0, paths=shared["paths"],
                horizon=shared["horizon"], mu=mu, sigma=sigma,
                chunk_size=shared["chunk_size"], seed=shared["seed"], cost=shared["cost"],
                drift_mode=shared["drift_mode"], manual_drift=shared["manual_drift"],
                historical_returns=market.daily_log_returns,
                stress_enabled=shared["stress_enabled"],
                stress_crash_pct=shared["stress_crash"],
                stress_vol_multiplier=shared["stress_vol_mult"],
                stress_drift_haircut=shared["stress_haircut"],
                variance_reduction=shared.get("variance_reduction", mc_core.VR_NONE),
                ruin_threshold=shared.get("ruin_threshold", 0.50),
                model=mc_core.MODEL_GBM,
            )
        except ValueError as exc:
            st.error(f"Invalid configuration: {exc}")
            st.stop()
        with st.spinner(
            f"Comparing {len(selected)} models on {shared['paths']:,} paths each..."
        ):
            report = mc_core.compare_models(
                base_config, models=selected, market=market,
                evt=shared.get("evt_enabled", False),
            )
        st.session_state["mc_comparison"] = report
        st.session_state["mc_comp_market"] = market

    report = st.session_state.get("mc_comparison")
    if report is None:
        st.write("Pick models and click **Run comparison**.")
        return

    cfg = report.base_config
    market = st.session_state.get("mc_comp_market")
    if market is not None and market.source == "fallback":
        st.warning(f"Market data unavailable; using fallback parameters. {market.note}")
    st.caption(
        f"Ticker {cfg.ticker} | {cfg.paths:,} paths | horizon {cfg.horizon} | "
        f"chunk {cfg.chunk_size:,} | seed {cfg.seed} | drift {cfg.drift_mode}"
    )

    # Comparison table.
    df = pd.DataFrame(report.rows).rename(columns=COMPARISON_DISPLAY_NAMES)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Most-conservative headline.
    if report.most_conservative:
        st.success(
            f"Most conservative model: **{report.most_conservative}** "
            "(highest probability of a >20% loss and highest 99% Expected Shortfall)."
        )

    if report.all_chunk_safe:
        st.info("All compared models ran chunk-safe (no full path x step matrix).")
    else:
        st.error("One or more models were NOT chunk-safe.")

    # Exports.
    st.subheader("Export comparison")
    csv_text = mc_core.comparison_to_csv(report)
    json_text = mc_core.comparison_to_json(report)
    e1, e2 = st.columns(2)
    e1.download_button(
        "Download CSV comparison", data=csv_text,
        file_name=f"{cfg.ticker}_model_comparison.csv", mime="text/csv",
    )
    e2.download_button(
        "Download JSON comparison", data=json_text,
        file_name=f"{cfg.ticker}_model_comparison.json", mime="application/json",
    )


def _render_investment_report(st, pd, shared: dict) -> None:
    """Render the institutional Investment Report tab for a non-coder investor."""
    import mc_report

    st.subheader("Investment Report")
    st.caption(
        "Runs the full institutional model stack, stress tests, a model-risk "
        "confidence score, and a benchmark comparison, then explains the results "
        "in plain English. This is a risk simulation, not investment advice."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.text_input("Ticker", value=shared["ticker"], key="ir_ticker").strip().upper()
        horizon = st.number_input("Investment horizon (trading days)", min_value=1,
                                  max_value=2520, value=int(shared["horizon"]), step=1,
                                  key="ir_horizon")
        paths = st.number_input("Paths", min_value=1_000, max_value=5_000_000,
                                value=int(shared["paths"]), step=10_000, key="ir_paths")
        chunk_size = st.number_input("Chunk size", min_value=1_000, max_value=200_000,
                                     value=int(shared["chunk_size"]), step=5_000,
                                     key="ir_chunk")
    with c2:
        risk_tolerance = st.selectbox("Risk tolerance", list(mc_report.RISK_TOLERANCES),
                                      index=1, key="ir_tol")
        max_loss = st.number_input("Maximum acceptable loss (%)", min_value=1.0,
                                   max_value=95.0, value=20.0, step=1.0, key="ir_maxloss")
        ruin = st.number_input("Ruin threshold (% of start)", min_value=5.0,
                               max_value=95.0, value=50.0, step=5.0, key="ir_ruin")
        amount = st.number_input("Investment amount", min_value=0.0,
                                 value=10_000.0, step=1_000.0, key="ir_amount")
    with c3:
        drift_mode = st.selectbox("Drift mode", list(mc_core.DRIFT_MODES),
                                  index=list(mc_core.DRIFT_MODES).index(shared["drift_mode"]),
                                  key="ir_drift")
        benchmark = st.text_input("Benchmark ticker", value=mc_report.DEFAULT_BENCHMARK,
                                  key="ir_bench").strip().upper()
        seed_text = st.text_input("Seed (blank = random)", value=str(shared["seed"] or ""),
                                  key="ir_seed")
        save_files = st.checkbox("Also write files to outputs/", value=False, key="ir_save")

    run_report = st.button("Run Report", type="primary", key="run_investment_report")

    if run_report:
        seed = int(seed_text) if seed_text.strip() else None
        rcfg = mc_report.InvestmentReportConfig(
            ticker=ticker or "ASSET", horizon=int(horizon), paths=int(paths),
            chunk_size=int(chunk_size), seed=seed, drift_mode=drift_mode,
            risk_tolerance=risk_tolerance, max_acceptable_loss_pct=max_loss / 100.0,
            ruin_threshold=ruin / 100.0, investment_amount=float(amount),
            benchmark=benchmark or mc_report.DEFAULT_BENCHMARK, years=shared["years"],
        )
        try:
            with st.spinner(
                f"Running {len(mc_report.INSTITUTIONAL_MODELS)} models + stress tests "
                f"for {ticker}..."
            ):
                report = mc_report.build_investment_report(rcfg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Report failed: {exc}")
            st.stop()
        st.session_state["mc_report"] = report
        if save_files:
            paths_written = mc_report.write_investment_report(report, outdir="outputs")
            st.session_state["mc_report_files"] = paths_written

    report = st.session_state.get("mc_report")
    if report is None:
        st.write("Set your inputs and click **Run Report**.")
        return

    # ---------------------- Risk cards ----------------------
    c = report["central"]
    st.markdown("### Plain-English risk cards")
    cards = st.columns(3)
    cards[0].metric("Profit Chance", f"{c['prob_profit']*100:.0f}%")
    cards[1].metric("Big Loss Chance (>20%)", f"{c['prob_loss_20']*100:.0f}%")
    cards[2].metric("Severe Drawdown Chance (50%)", f"{c['prob_drawdown_50']*100:.0f}%")
    cards2 = st.columns(3)
    cards2[0].metric("Worst Model", report["worst_model"])
    cards2[1].metric("Model Confidence", report["model_confidence"])
    cards2[2].metric("Investment Label", report["investment_label"])

    if report["model_confidence"] == mc_report.CONFIDENCE_LOW:
        st.error("Model confidence is LOW. Do not rely on this result alone.")

    # ---------------------- Plain-English sections ----------------------
    for name in mc_report.REPORT_SECTIONS:
        st.markdown(f"#### {name}")
        st.markdown(report["plain_english"].get(name, ""))

    # ---------------------- Stress tests ----------------------
    st.markdown("#### Stress test results")
    st.dataframe(pd.DataFrame(report["stress_tests"]["rows"]),
                 hide_index=True, use_container_width=True)

    # ---------------------- Benchmark ----------------------
    bm = report["benchmark_comparison"]
    st.markdown("#### Benchmark comparison")
    if bm.get("available"):
        st.dataframe(pd.DataFrame([{
            "Benchmark": bm["benchmark"],
            "Excess return": bm["excess_return"],
            "Beta": bm["beta"],
            "Correlation": bm["correlation"],
            "Ticker max DD": bm["ticker_max_drawdown"],
            "Benchmark max DD": bm["benchmark_max_drawdown"],
        }]), hide_index=True, use_container_width=True)
    else:
        st.info("Benchmark comparison unavailable.")

    # ---------------------- Fundamentals ----------------------
    st.markdown("#### Fundamentals (sanity check)")
    f = report["fundamentals"]
    if f.get("available"):
        st.json(f["fields"])
    else:
        st.info(f.get("note", "Fundamental data unavailable from source."))

    # ---------------------- Model-risk warnings ----------------------
    if report["model_risk"]["warnings"]:
        st.markdown("#### Model risk warnings")
        for wmsg in report["model_risk"]["warnings"]:
            st.warning(wmsg)

    # ---------------------- Exports ----------------------
    st.markdown("#### Export")
    md_text = mc_report.render_markdown(report)
    json_text = mc_report.report_to_json(report)
    csv_text = mc_report.comparison_csv(report)
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download Markdown report", data=md_text,
                       file_name=f"{report['inputs']['ticker']}_investment_report.md",
                       mime="text/markdown")
    e2.download_button("Download JSON report", data=json_text,
                       file_name=f"{report['inputs']['ticker']}_investment_report.json",
                       mime="application/json")
    e3.download_button("Download comparison CSV", data=csv_text,
                       file_name=f"{report['inputs']['ticker']}_institutional_comparison.csv",
                       mime="text/csv")
    if st.session_state.get("mc_report_files"):
        st.success(f"Files written: {st.session_state['mc_report_files']}")

    st.caption(report["disclaimer"])


def _render_portfolio(st, pd, shared: dict) -> None:
    """Render the Portfolio mode tab: multi-asset correlated simulation."""
    st.subheader("Portfolio mode (multi-asset, laptop-safe)")
    st.caption(
        "Simulate a correlated multi-asset GBM portfolio. Historical returns are "
        "fetched per ticker, covariance is shrunk, and paths are generated in a "
        "chunk-safe per-asset block (no paths x steps matrix)."
    )

    tickers_text = st.text_input(
        "Tickers (comma-separated)", value="AAPL,MSFT",
        help="Example: AAPL,MSFT,NVDA",
    )
    weights_text = st.text_input(
        "Weights (comma-separated, blank = equal weight)", value="",
    )
    run_pf = st.button("Run portfolio", type="primary", key="run_portfolio")

    if run_pf:
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Enter at least two tickers.")
            st.stop()
        returns_by_ticker = {}
        s0_by_ticker = {}
        with st.spinner(f"Fetching history for {', '.join(tickers)}..."):
            for t in tickers:
                mp = mc_core.estimate_parameters_from_history(t, years=shared["years"])
                returns_by_ticker[t] = mp.daily_log_returns
                s0_by_ticker[t] = mp.s0
        weights = None
        if weights_text.strip():
            try:
                weights = [float(w) for w in weights_text.split(",")]
            except ValueError:
                st.error("Weights must be numbers.")
                st.stop()
        try:
            pf = mc_core.simulate_portfolio(
                returns_by_ticker, weights=weights, s0_by_ticker=s0_by_ticker,
                paths=shared["paths"], horizon=shared["horizon"],
                chunk_size=shared["chunk_size"], seed=shared["seed"],
                drift_mode=shared["drift_mode"], manual_drift=shared["manual_drift"],
            )
        except ValueError as exc:
            st.error(f"Portfolio error: {exc}")
            st.stop()
        st.session_state["mc_portfolio"] = pf

    pf = st.session_state.get("mc_portfolio")
    if pf is None:
        st.write("Enter tickers and click **Run portfolio**.")
        return

    ps = pf["statistics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected portfolio value", f"{ps['expected_value']:.4f}",
              f"{ps['expected_return']:+.2%}")
    c2.metric("Median portfolio value", f"{ps['median_value']:.4f}")
    c3.metric("VaR 99%", f"{ps['var']['99']['value']:.4f}")
    c4.metric("ES 99%", f"{ps['expected_shortfall']['99']['value']:.4f}")
    st.caption(
        f"Covariance method: {pf['covariance_method']} | "
        f"Cholesky jittered: {pf['cholesky_jittered']} | "
        f"chunk-safe: {pf['chunk_safe']}"
    )

    st.markdown("**Per-asset summary**")
    per_asset_df = pd.DataFrame(pf["per_asset"]).T
    st.dataframe(per_asset_df, use_container_width=True)

    st.markdown("**Correlation matrix**")
    corr_df = pd.DataFrame(
        pf["correlation_matrix"], index=pf["tickers"], columns=pf["tickers"]
    )
    st.dataframe(corr_df, use_container_width=True)

    import json as _json
    e1, e2 = st.columns(2)
    e1.download_button(
        "Download correlation CSV", data=mc_core.portfolio_correlation_csv(pf),
        file_name="portfolio_correlation.csv", mime="text/csv",
    )
    pf_export = {k: v for k, v in pf.items() if k != "portfolio_values"}
    e2.download_button(
        "Download portfolio JSON",
        data=_json.dumps(pf_export, indent=2, default=mc_core._json_default),
        file_name="portfolio_report.json", mime="application/json",
    )


def _render_tactical(st, pd, shared: dict) -> None:
    """Phase 2 short-horizon tactical trading-rule simulator tab."""
    from tactical_config import TradingRule, preset_5_day, preset_10_day
    from tactical_simulator import run_tactical_simulation

    st.subheader("Tactical short-horizon rule tester (5–10 trading days)")
    st.caption(
        "Simulate many short price paths with the existing Monte Carlo engine, "
        "then apply a trading rule (entry, stop, take-profit, trailing stop, "
        "max hold, optional re-entry). Results are research stats — not live signals."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        t_ticker = st.text_input("Tactical ticker", value=shared.get("ticker") or "AAPL",
                                 key="tac_ticker").strip().upper()
        t_horizon = st.selectbox("Horizon (trading days)", [5, 10], index=0, key="tac_h")
        t_paths = st.number_input("Paths", min_value=1_000, max_value=200_000,
                                  value=20_000, step=1_000, key="tac_paths")
    with c2:
        t_side = st.selectbox("Side", ["long", "short"], index=0, key="tac_side")
        t_stop = st.number_input("Stop loss %", min_value=0.0, max_value=50.0,
                                 value=2.0, step=0.25, key="tac_stop") / 100.0
        t_tp_on = st.checkbox("Enable take-profit", value=True, key="tac_tp_on")
        t_tp = st.number_input("Take profit %", min_value=0.1, max_value=50.0,
                               value=3.0, step=0.25, key="tac_tp") / 100.0 if t_tp_on else None
    with c3:
        t_trail_on = st.checkbox("Enable trailing stop", value=False, key="tac_trail_on")
        t_trail = (
            st.number_input("Trailing stop %", min_value=0.1, max_value=50.0,
                            value=1.5, step=0.25, key="tac_trail") / 100.0
            if t_trail_on else None
        )
        t_reentry = st.checkbox("Allow re-entry", value=False, key="tac_re")
        t_max_trades = st.number_input("Max trades / path", min_value=1, max_value=10,
                                       value=1, key="tac_mt") if t_reentry else 1
        t_cost = st.number_input("Cost fraction", min_value=0.0, max_value=0.05,
                                 value=0.001, step=0.0005, format="%.4f", key="tac_cost")
        t_seed = st.number_input("Seed", min_value=0, value=42, step=1, key="tac_seed")

    t_hist = st.checkbox("Also run historical rolling windows", value=False, key="tac_hist")
    t_var_bt = st.checkbox("Rolling VaR coverage (Kupiec)", value=False, key="tac_var")
    t_vr = st.selectbox(
        "Variance reduction (MC engine)",
        ["none", "antithetic", "sobol", "control_variate"],
        index=0,
        key="tac_vr",
    )

    run_tac = st.button("Run tactical simulation", type="primary", key="tac_run")

    if run_tac:
        progress = st.progress(0, text="Building config…")
        try:
            base = preset_5_day if int(t_horizon) <= 5 else preset_10_day
            cfg = base(
                t_ticker,
                paths=int(t_paths),
                seed=int(t_seed),
                transaction_cost=float(t_cost),
                horizon_days=int(t_horizon),
            )
            rule = TradingRule(
                name=f"GUI {t_side} {t_horizon}d",
                entry_condition=f"Enter {t_side} at start",
                exit_condition="Exit on stop/TP/trail/max hold",
                stop_loss_pct=float(t_stop),
                max_holding_days=int(t_horizon),
                side=t_side,
                take_profit_pct=t_tp,
                trailing_stop_pct=t_trail,
                allow_reentry=bool(t_reentry),
                max_trades=int(t_max_trades),
            )
            cfg = cfg.with_rule(rule)
            progress.progress(30, text="Generating paths + applying rule…")

            hist_px = None
            if t_hist or t_var_bt:
                mkt = mc_core.estimate_parameters_from_history(t_ticker)
                if mkt.daily_log_returns is not None and mkt.daily_log_returns.size > 2:
                    lr = np.asarray(mkt.daily_log_returns, dtype=float)
                    px = float(mkt.s0) * np.exp(np.cumsum(np.r_[0.0, lr]))
                    hist_px = px * (float(mkt.s0) / px[-1])

            result = run_tactical_simulation(
                cfg,
                historical_prices=hist_px if t_hist else None,
                run_var_backtest=bool(t_var_bt),
                variance_reduction=t_vr,
            )
            progress.progress(100, text="Done")
            st.session_state["tac_result"] = result
        except Exception as exc:  # noqa: BLE001
            st.error(f"Tactical run failed: {exc}")
            progress.empty()
            return

    result = st.session_state.get("tac_result")
    if result is None:
        st.info("Configure the rule above and click **Run tactical simulation**.")
        return

    s = result.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profit chance", f"{s['prob_profit']:.2%}")
    m2.metric("Avg P&L", f"{s['avg_pnl']:+.4f}")
    m3.metric("Worst P&L", f"{s['worst_pnl']:+.4f}")
    m4.metric("Stop hit rate", f"{s['stop_hit_rate']:.2%}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("TP rate", f"{s.get('take_profit_rate', 0):.2%}")
    m6.metric("Trail rate", f"{s.get('trailing_stop_rate', 0):.2%}")
    m7.metric("Avg trades/path", f"{s['avg_trades_per_path']:.2f}")
    m8.metric("Avg hold (days)", f"{s['avg_holding_days']:.2f}")

    st.markdown("**Summary**")
    st.code(result.summary_text())

    # P&L histogram (matplotlib via streamlit)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(result.pnl, bins=60, color="#3b7dd8", alpha=0.85)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{result.config.ticker}: tactical P&L distribution")
        ax.set_xlabel("P&L per share")
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Chart skipped: {exc}")

    # Sample paths
    try:
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        n_show = min(40, result.price_paths.shape[0])
        for i in range(n_show):
            ax2.plot(result.price_paths[i], linewidth=0.7, alpha=0.6)
        ax2.set_title(f"Sample paths ({n_show} of {result.price_paths.shape[0]:,})")
        ax2.set_xlabel("trading day")
        ax2.set_ylabel("price")
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception:
        pass

    stats_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in result.to_stats_dict().items()
         if not isinstance(v, (list, dict))]
    )
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    import json as _json
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download tactical stats JSON",
        data=_json.dumps(result.to_stats_dict(), indent=2, default=str),
        file_name=f"{result.config.ticker}_tactical_stats.json",
        mime="application/json",
        key="tac_dl_json",
    )
    d2.download_button(
        "Download P&L CSV",
        data=pd.DataFrame({"pnl": result.pnl, "n_trades": result.n_trades,
                           "stop_hit": result.stop_hit}).to_csv(index=False),
        file_name=f"{result.config.ticker}_tactical_pnl.csv",
        mime="text/csv",
        key="tac_dl_csv",
    )


if __name__ == "__main__":
    main()
