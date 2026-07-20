import json
from pathlib import Path

import pytest

from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.options_market import (
    EquityQuote,
    OptionContract,
    OptionQuote,
    parse_equity_quotes,
    parse_option_instruments,
    parse_option_quotes,
    strike_band,
)
from engine.robinhood.options_strategy import decide_order
from engine.robinhood.options_market import SymbolMarketSnapshot

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("RH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RH_OPTIONS_MIN_DTE", "0")
    monkeypatch.setenv("RH_OPTIONS_MAX_DTE", "60")
    monkeypatch.setenv("RH_OPTIONS_MAX_PREMIUM_USD", "500")
    return RobinhoodConfig.from_env()


def test_parse_equity_quotes_fixture():
    payload = json.loads((FIXTURES / "equity_quotes_spy.json").read_text())
    out = parse_equity_quotes(payload, ["SPY"])
    assert "SPY" in out
    assert out["SPY"].last_price == pytest.approx(500.25)


def test_parse_option_instruments_fixture(cfg):
    payload = json.loads((FIXTURES / "option_instruments_spy.json").read_text())
    # The fixture's hardcoded expiration went stale on 2026-07-19 and the
    # parser (correctly) dropped the expired contracts. Pin expirations to a
    # rolling future date so the test exercises parsing, not the calendar.
    from datetime import date, timedelta

    future = (date.today() + timedelta(days=10)).isoformat()
    for row in payload:
        row["expiration_date"] = future
    lo, hi = strike_band(500.0, cfg.options_strike_band_pct)
    calls = parse_option_instruments(
        payload,
        underlying="SPY",
        min_dte=0,
        max_dte=60,
        strike_lo=lo,
        strike_hi=hi,
        option_type="call",
    )
    assert len(calls) == 2
    assert all(c.option_type == "call" for c in calls)


def test_parse_option_quotes_spread():
    payload = json.loads((FIXTURES / "option_quotes_spy.json").read_text())
    out = parse_option_quotes(payload)
    q = out["opt-spy-call-505"]
    assert q.mid == pytest.approx(2.5)
    assert q.spread_pct == pytest.approx(8.0, rel=0.1)


def test_decide_order_call_otm(cfg):
    spot = EquityQuote(symbol="SPY", last_price=500.25)
    contracts = [
        OptionContract(
            symbol="SPY",
            option_type="call",
            strike=500.0,
            expiration_date="2026-07-18",
            instrument_id="opt-spy-call-500",
            dte=10,
        ),
        OptionContract(
            symbol="SPY",
            option_type="call",
            strike=505.0,
            expiration_date="2026-07-18",
            instrument_id="opt-spy-call-505",
            dte=10,
        ),
    ]
    quotes = {
        "opt-spy-call-505": OptionQuote(
            instrument_id="opt-spy-call-505",
            bid=2.4,
            ask=2.6,
            mid=2.5,
            spread_pct=8.0,
        )
    }
    snap = SymbolMarketSnapshot(symbol="SPY", spot=spot, contracts=contracts, quotes=quotes)
    intent, stage = decide_order(snap, cfg, "call")
    assert stage == "intent_ready"
    assert intent is not None
    assert intent.instrument_id == "opt-spy-call-505"
    assert intent.option_type == "call"
    assert "place_option_order" not in intent.to_mcp_args()  # sanity — dict keys only
    assert intent.to_mcp_args()["side"] == "buy"
