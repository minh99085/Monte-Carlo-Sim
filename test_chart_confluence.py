"""Ground-truth tests for the chart-battery confluence rules."""

from __future__ import annotations

from chart_confluence import MIN_CONFIDENCE, combine


def _c(role, bias="bullish", conf=0.8, rsi=None, ticker="NVDA"):
    return {"role": role, "bias": bias, "confidence": conf,
            "rsi": rsi, "ticker": ticker}


def test_aligned_full_size():
    out = combine([_c("weekly"), _c("daily", rsi=60.0)])
    assert out["stance"] == "ALIGNED_LONG"
    assert out["allowed"] is True
    assert out["size_multiplier"] == 1.0


def test_missing_daily_is_incomplete():
    out = combine([_c("weekly")])
    assert out["stance"] == "INCOMPLETE"
    assert out["allowed"] is False
    assert out["size_multiplier"] == 0.0


def test_weekly_bearish_blocks_everything():
    out = combine([_c("weekly", bias="bearish"), _c("daily")])
    assert out["stance"] == "NO_TRADE"
    assert out["allowed"] is False


def test_daily_disagrees_means_wait():
    out = combine([_c("weekly"), _c("daily", bias="bearish")])
    assert out["stance"] == "MIXED"
    assert out["allowed"] is False


def test_neutral_weekly_halves_size():
    out = combine([_c("weekly", bias="neutral"), _c("daily")])
    assert out["allowed"] is True
    assert out["size_multiplier"] == 0.5


def test_low_confidence_refuses():
    out = combine([_c("weekly", conf=MIN_CONFIDENCE - 0.01), _c("daily")])
    assert out["stance"] == "NO_TRADE"
    assert "confidence" in out["reasons"][0]


def test_overbought_rsi_halves():
    out = combine([_c("weekly"), _c("daily", rsi=80.0)])
    assert out["allowed"] is True
    assert out["size_multiplier"] == 0.5
    assert any("overbought" in w for w in out["warnings"])


def test_ratio_bearish_halves_ratio_bullish_keeps():
    lagging = combine([_c("weekly"), _c("daily"),
                       _c("ratio", bias="bearish", ticker="NVDA/SPY")])
    assert lagging["allowed"] is True
    assert lagging["size_multiplier"] == 0.5
    leading = combine([_c("weekly"), _c("daily"),
                       _c("ratio", bias="bullish", ticker="NVDA/SPY")])
    assert leading["size_multiplier"] == 1.0


def test_penalties_stack():
    out = combine([_c("weekly", bias="neutral"), _c("daily", rsi=80.0),
                   _c("ratio", bias="bearish", ticker="NVDA/SPY")])
    assert out["allowed"] is True
    assert out["size_multiplier"] == 0.125  # 0.5 * 0.5 * 0.5 — every flag stacks


def test_ticker_mismatch_refuses():
    out = combine([_c("weekly", ticker="NVDA"), _c("daily", ticker="AAPL")])
    assert out["stance"] == "NO_TRADE"
    assert "not the same asset" in out["reasons"][0]


def test_ratio_ticker_exempt_from_mismatch():
    out = combine([_c("weekly"), _c("daily"),
                   _c("ratio", ticker="NVDA/SPY")])
    assert out["stance"] == "ALIGNED_LONG"


def test_garbage_bias_treated_as_neutral():
    out = combine([_c("weekly", bias="TO THE MOON"), _c("daily")])
    # Unknown weekly bias falls back to neutral → half size, not a crash.
    assert out["allowed"] is True
    assert out["size_multiplier"] == 0.5


# ---------------------------------------------------------------------------
# Action verbs (flat path)
# ---------------------------------------------------------------------------


def test_flat_actions_map_to_stances():
    assert combine([_c("weekly"), _c("daily")])["action"] == "BUY"
    assert combine([_c("weekly"), _c("daily", bias="neutral")])["action"] == "WAIT"
    assert combine([_c("weekly", bias="bearish"), _c("daily")])["action"] == "AVOID"
    assert combine([_c("weekly")])["action"] == "UPLOAD_MORE"


# ---------------------------------------------------------------------------
# Holding path: HOLD / SELL (never a fresh BUY, never a short)
# ---------------------------------------------------------------------------

POS = {"symbol": "NVDA", "qty": 10, "entry_price": 100.0}


def test_holding_thesis_intact_holds():
    out = combine([_c("weekly"), _c("daily")], position=POS)
    assert out["stance"] == "HOLDING" and out["action"] == "HOLD"
    assert out["allowed"] is False


def test_holding_weekly_bearish_sells():
    out = combine([_c("weekly", bias="bearish"), _c("daily")], position=POS)
    assert out["stance"] == "EXIT" and out["action"] == "SELL"
    assert "tide" in out["reasons"][0]


def test_holding_deterioration_sells():
    out = combine([_c("weekly", bias="neutral"), _c("daily", bias="bearish"),
                   _c("ratio", bias="bearish", ticker="NVDA/SPY")],
                  position=POS)
    assert out["action"] == "SELL"
    assert "deteriorating" in out["reasons"][0]


def test_holding_daily_bearish_alone_holds_with_warning():
    out = combine([_c("weekly"), _c("daily", bias="bearish"),
                   _c("ratio", bias="bullish", ticker="NVDA/SPY")],
                  position=POS)
    assert out["action"] == "HOLD"
    assert out["warnings"]


def test_holding_low_confidence_never_exits():
    out = combine([_c("weekly", bias="bearish", conf=0.3), _c("daily")],
                  position=POS)
    assert out["action"] == "HOLD"          # bad read must not trigger a sell
    assert "unreliable" in out["reasons"][0]


def test_holding_wrong_symbol_charts_hold():
    out = combine([_c("weekly", ticker="AAPL"), _c("daily", ticker="AAPL")],
                  position=POS)
    assert out["action"] == "HOLD"
    assert "not applied" in out["reasons"][0]


def test_holding_incomplete_holds():
    out = combine([_c("weekly")], position=POS)
    assert out["action"] == "HOLD"
    assert "missing" in out["reasons"][0]
