"""Acceptance tests for signal_calibration.py (Phase A)."""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import signal_calibration as sc


# ---------------------------------------------------------------------------
# Reference implementations (independent, loop-based, for fixture values)
# ---------------------------------------------------------------------------


def ref_wilder_rsi(prices, length=14):
    """Textbook Wilder RSI: SMA-seeded average gain/loss, then
    avg = (prev * (len-1) + current) / len. Independent of the module's
    vectorized/rma-based implementation."""
    p = [float(x) for x in prices]
    diffs = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    gains = [max(d, 0.0) for d in diffs]
    losses = [max(-d, 0.0) for d in diffs]
    out = [float("nan")] * len(p)
    if len(diffs) < length:
        return out
    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length

    def to_rsi(g, l):
        if l == 0.0:
            return 100.0 if g > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + g / l)

    out[length] = to_rsi(avg_gain, avg_loss)
    for i in range(length, len(diffs)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        out[i + 1] = to_rsi(avg_gain, avg_loss)
    return out


# Fixture: the classic 14-period RSI worked example (StockCharts/Wilder-style
# daily closes). Expected values computed with ref_wilder_rsi above.
RSI_FIXTURE_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
    45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
    46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
    45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
]


class TestRSI:
    def test_matches_reference_fixture(self):
        got = sc.rsi(np.array(RSI_FIXTURE_CLOSES), 14)
        want = ref_wilder_rsi(RSI_FIXTURE_CLOSES, 14)
        for i in range(len(RSI_FIXTURE_CLOSES)):
            if math.isnan(want[i]):
                assert math.isnan(got[i]), f"index {i}: expected NaN"
            else:
                assert got[i] == pytest.approx(want[i], abs=1e-9), f"index {i}"

    def test_first_defined_value_of_fixture(self):
        # Hand-checkable: the first RSI of the classic fixture is ~70.46.
        got = sc.rsi(np.array(RSI_FIXTURE_CLOSES), 14)
        assert got[14] == pytest.approx(70.46, abs=0.1)

    def test_matches_reference_on_random_walks(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=120)))
            got = sc.rsi(prices, 14)
            want = ref_wilder_rsi(prices, 14)
            np.testing.assert_allclose(got[14:], want[14:], atol=1e-9)

    def test_monotonic_extremes(self):
        up = np.linspace(100, 150, 40)
        down = np.linspace(150, 100, 40)
        assert sc.rsi(up, 14)[-1] == pytest.approx(100.0)
        assert sc.rsi(down, 14)[-1] == pytest.approx(0.0)

    def test_warmup_is_nan(self):
        vals = sc.rsi(np.array(RSI_FIXTURE_CLOSES), 14)
        assert np.all(np.isnan(vals[:14]))
        assert np.isfinite(vals[14])


class TestEMA:
    def test_sma_seed_and_recursion(self):
        x = np.arange(1.0, 11.0)  # 1..10
        e = sc.ema(x, 3)
        assert np.all(np.isnan(e[:2]))
        assert e[2] == pytest.approx(2.0)  # SMA(1,2,3)
        alpha = 0.5  # 2/(3+1)
        expect = 2.0
        for i in range(3, 10):
            expect = alpha * x[i] + (1 - alpha) * expect
            assert e[i] == pytest.approx(expect)

    def test_constant_series(self):
        e = sc.ema(np.full(50, 42.0), 9)
        assert np.allclose(e[8:], 42.0)


class TestBuckets:
    def test_edges(self):
        assert sc.bucket("bullish", 39.999) == "bullish_low"
        assert sc.bucket("bullish", 40.0) == "bullish_mid"
        assert sc.bucket("bullish", 60.0) == "bullish_mid"
        assert sc.bucket("bullish", 60.001) == "bullish_high"
        assert sc.bucket("bearish", 10.0) == "bearish_low"
        assert sc.bucket("BEARISH", 75.0) == "bearish_high"

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            sc.bucket("sideways", 50.0)
        with pytest.raises(ValueError):
            sc.bucket("bullish", float("nan"))

    def test_all_buckets_constant(self):
        assert len(sc.ALL_BUCKETS) == 6


class TestShrinkage:
    def test_goes_to_zero_as_n_eff_goes_to_zero(self):
        raw = 0.01
        prev = abs(sc.shrink_mu(raw, 1000.0, t_stat=5.0))
        for n_eff in (300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3):
            cur = abs(sc.shrink_mu(raw, n_eff, t_stat=5.0))
            assert cur <= prev + 1e-15
            prev = cur
        assert sc.shrink_mu(raw, 1e-9, t_stat=5.0) == pytest.approx(0.0, abs=1e-10)
        assert sc.shrink_mu(raw, 0.0, t_stat=5.0) == 0.0

    def test_hard_zero_below_t_one(self):
        assert sc.shrink_mu(0.05, 500.0, t_stat=0.99) == 0.0
        assert sc.shrink_mu(-0.05, 500.0, t_stat=-0.5) == 0.0
        assert sc.shrink_mu(0.05, 500.0, t_stat=float("nan")) == 0.0
        assert sc.shrink_mu(0.05, 500.0, t_stat=1.5) != 0.0

    def test_retains_most_only_with_strong_evidence(self):
        # For a well-populated bucket retention -> t^2 / (t^2 + T_PRIOR):
        # exactly 50% at |t| = 2 — "most" of the mean needs |t| >~ 2.
        big_n = 1e9
        assert sc.shrink_mu(0.01, big_n, 2.0) == pytest.approx(0.005, rel=1e-6)
        assert sc.shrink_mu(0.01, big_n, 3.0) == pytest.approx(
            0.01 * 9 / 13, rel=1e-6)
        # A |t| just past the hard-zero gate keeps well under a third even
        # with unlimited samples.
        assert sc.shrink_mu(0.01, big_n, 1.2) < 0.01 / 3

    def test_retention_monotone_in_t(self):
        vals = [abs(sc.shrink_mu(0.01, 500.0, t))
                for t in (1.1, 1.5, 2.0, 3.0, 5.0)]
        assert vals == sorted(vals)


class TestNeweyWest:
    def test_nw_se_exceeds_naive_on_autocorrelated_data(self):
        # Overlapping 5-day sums of iid noise -> positive autocorrelation
        # up to lag 4, exactly the structure of overlapping forward returns.
        rng = np.random.default_rng(123)
        daily = rng.normal(0.0, 0.01, size=3000)
        h = 5
        overlapping = np.convolve(daily, np.ones(h), mode="valid")
        st = sc.newey_west_stats(overlapping, lag=h - 1)
        assert st["se_nw"] > 1.5 * st["se_naive"]
        assert st["n_eff"] < 0.6 * st["n"]

    def test_iid_data_close_to_naive(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0.0, 1.0, size=5000)
        st = sc.newey_west_stats(x, lag=4)
        assert st["se_nw"] == pytest.approx(st["se_naive"], rel=0.15)
        assert st["n_eff"] == pytest.approx(st["n"], rel=0.2)

    def test_degenerate_sizes(self):
        st = sc.newey_west_stats(np.array([0.01]), lag=4)
        assert math.isnan(st["t_stat"])
        assert sc.shrink_mu(0.01, st["n_eff"], st["t_stat"]) == 0.0


class TestCalibrateAndPersistence:
    def _synthetic_prices(self, n=2600, seed=11, mu_daily=0.0, sigma_daily=0.012):
        rng = np.random.default_rng(seed)
        rets = rng.normal(mu_daily, sigma_daily, size=n)
        return 100.0 * np.exp(np.cumsum(np.r_[0.0, rets]))

    def test_calibrate_offline_structure(self):
        prices = self._synthetic_prices()
        table = sc.calibrate("TEST", years=10.0, horizon_days=5, prices=prices)
        assert set(table.buckets) == set(sc.ALL_BUCKETS)
        total_n = sum(b.n for b in table.buckets.values())
        # Nearly every post-warm-up day should land in exactly one bucket.
        assert total_n > len(prices) - 50
        for b in table.buckets.values():
            if b.n >= 2:
                assert b.se_weekly > 0
                assert b.shrunk_mu_annual == pytest.approx(
                    b.shrunk_mu_weekly * 252 / 5)
                # shrinkage never amplifies
                assert abs(b.shrunk_mu_weekly) <= abs(b.raw_mu_weekly) + 1e-15

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        prices = self._synthetic_prices()
        table = sc.calibrate("RT", years=10.0, horizon_days=5, prices=prices)
        path = table.save(tmp_path)
        assert path.name == "RT_5d.json"
        loaded = sc.load_calibration("RT", 5, tmp_path)
        assert loaded.ticker == "RT"
        for name in sc.ALL_BUCKETS:
            assert loaded.buckets[name].raw_mu_weekly == pytest.approx(
                table.buckets[name].raw_mu_weekly)
        payload = json.loads(path.read_text())
        assert "created_at" in payload
        assert "annualization" in payload

    def test_staleness_warning_and_error(self, tmp_path: Path):
        prices = self._synthetic_prices(n=300)
        old = datetime.now(timezone.utc) - timedelta(days=45)
        table = sc.calibrate("STALE", years=1.0, horizon_days=5,
                             prices=prices, now=old)
        table.save(tmp_path)
        with pytest.warns(UserWarning, match="days old"):
            sc.load_calibration("STALE", 5, tmp_path)

        ancient = datetime.now(timezone.utc) - timedelta(days=200)
        table2 = sc.calibrate("DEAD", years=1.0, horizon_days=5,
                              prices=prices, now=ancient)
        table2.save(tmp_path)
        with pytest.raises(sc.CalibrationStaleError):
            sc.load_calibration("DEAD", 5, tmp_path)

    def test_fresh_load_no_warning(self, tmp_path: Path):
        prices = self._synthetic_prices(n=300)
        sc.calibrate("FRESH", years=1.0, horizon_days=5,
                     prices=prices).save(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sc.load_calibration("FRESH", 5, tmp_path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            sc.load_calibration("NOPE", 5, tmp_path)

    def test_cli_writes_json(self, tmp_path: Path, monkeypatch):
        prices = self._synthetic_prices()

        def fake_download(ticker, years=8.0):
            idx = pd.bdate_range("2016-01-04", periods=len(prices))
            return prices, idx

        monkeypatch.setattr(sc, "download_history", fake_download)
        rc = sc.main(["FAKE", "--years", "8",
                      "--calibration-dir", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "FAKE_5d.json").is_file()


class TestComputeFeatures:
    def test_columns_and_warmup(self):
        rng = np.random.default_rng(3)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        df = sc.compute_features(prices)
        assert list(df.columns) == [
            "close", "ema_fast", "ema_slow", "trend", "rsi", "bucket"]
        assert pd.isna(df["trend"].iloc[10])      # slow EMA not seeded yet
        assert df["trend"].iloc[30] in ("bullish", "bearish")
        assert df["bucket"].iloc[30] in sc.ALL_BUCKETS
        # trend consistent with EMA comparison
        row = df.iloc[60]
        expected = "bullish" if row["ema_fast"] > row["ema_slow"] else "bearish"
        assert row["trend"] == expected
