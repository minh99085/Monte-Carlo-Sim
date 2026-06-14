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

        with st.expander("Advanced parameter overrides"):
            mu_override = st.text_input("mu override (annual, blank = estimate)", value="")
            sigma_override = st.text_input("sigma override (annual, blank = estimate)", value="")

        run_clicked = st.button("Run simulation", type="primary")

    seed = int(seed_text) if seed_text.strip() else None

    # ---------------------- Memory pre-flight ----------------------
    preview_cfg = mc_core.SimulationConfig(
        paths=int(paths), horizon=int(horizon), chunk_size=int(chunk_size)
    )
    preview_mem = mc_core.predict_memory(preview_cfg)
    st.info(preview_mem.status())

    # ------------------------------------------------------------------
    # Run only when the button is clicked; persist the result in
    # session_state so later reruns (e.g. clicking a download button) keep
    # showing the results instead of resetting to the initial screen.
    # ------------------------------------------------------------------
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

    # Render from the stored result so downloads/reruns never lose it.
    result = st.session_state.get("mc_result")
    market = st.session_state.get("mc_market")
    if result is None:
        st.write("Configure inputs in the sidebar and click **Run simulation**.")
        return

    config = result.config
    s = result.stats
    paths = config.paths
    chunk_size = config.chunk_size

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
    json_text = mc_core.report_to_json(result, market)
    e1, e2 = st.columns(2)
    e1.download_button(
        "Download CSV summary", data=csv_text,
        file_name=f"{ticker}_mc_summary.csv", mime="text/csv",
    )
    e2.download_button(
        "Download JSON report", data=json_text,
        file_name=f"{ticker}_mc_report.json", mime="application/json",
    )


if __name__ == "__main__":
    main()
