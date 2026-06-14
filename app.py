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
) -> mc_core.SimulationConfig:
    """Assemble a validated :class:`mc_core.SimulationConfig` from GUI inputs."""

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

    st.set_page_config(page_title="Monte Carlo GBM Simulator", layout="wide")
    st.title("Monte Carlo GBM Simulator")
    st.caption(
        "CPU-first, memory-safe Geometric Brownian Motion simulation. "
        "Never allocates a full path x step matrix."
    )

    # ---------------------- Sidebar inputs ----------------------
    with st.sidebar:
        st.header("Inputs")
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
        default_chunk = min(mc_core.DEFAULT_SERIOUS_CHUNK, int(paths))
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

    if not run_clicked:
        st.write("Configure inputs in the sidebar and click **Run simulation**.")
        return

    # ---------------------- Resolve market parameters ----------------------
    with st.spinner(f"Fetching parameters for {ticker}..."):
        market = mc_core.estimate_parameters_from_history(
            ticker, years=years,
            s0_override=(s0_override if s0_override > 0 else None),
        )
    mu = float(mu_override) if mu_override.strip() else market.mu
    sigma = float(sigma_override) if sigma_override.strip() else market.sigma

    if market.source == "fallback":
        st.warning(f"Market data unavailable; using fallback parameters. {market.note}")
    else:
        st.success(
            f"Estimated from {market.source}: S0={market.s0:,.2f}, "
            f"mu={mu:.2%}, sigma={sigma:.2%}"
        )

    config = build_config_from_inputs(
        ticker=ticker, s0=market.s0, paths=int(paths), horizon=int(horizon),
        mu=mu, sigma=sigma, chunk_size=int(chunk_size), seed=seed, cost=cost,
    )

    # ---------------------- Run ----------------------
    with st.spinner(f"Simulating {paths:,} paths in chunks of {chunk_size:,}..."):
        result = mc_core.simulate(config)

    s = result.stats

    # ---------------------- Headline metrics ----------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected ending value", f"{s['expected_value']:,.2f}",
              f"{s['expected_return']:+.2%}")
    c2.metric("Median ending value", f"{s['median_value']:,.2f}")
    c3.metric("Probability of profit", f"{s['prob_profit']:.2%}")
    c4.metric("Probability of loss", f"{s['prob_loss']:.2%}")

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
