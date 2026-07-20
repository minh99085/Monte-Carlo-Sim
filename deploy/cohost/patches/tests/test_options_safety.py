import pytest

from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.safety_gates import RobinhoodSafetyGates


@pytest.fixture
def gates(tmp_path, monkeypatch):
    monkeypatch.setenv("RH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RH_LIVE_TRADING_ENABLED", "1")
    monkeypatch.setenv("RH_OPTIONS_WATCHLIST", "SPY,QQQ")
    # Option notional now correctly includes the 100x contract multiplier, so
    # these $200-$1000 test orders would trip the generic per-order cap before
    # reaching the option-specific gates under test. Raise the cap so each
    # specific gate (watchlist / contracts / premium / long-only) is exercised.
    monkeypatch.setenv("RH_MAX_ORDER_NOTIONAL_USD", "2000")
    monkeypatch.setenv("RH_OPTIONS_MAX_CONTRACTS", "2")
    monkeypatch.setenv("RH_OPTIONS_MAX_PREMIUM_USD", "300")
    cfg = RobinhoodConfig.from_env()
    return RobinhoodSafetyGates(cfg)


def test_option_watchlist_block(gates):
    v = gates.evaluate(
        "place_option_order",
        {
            "symbol": "TSLA",
            "quantity": 1,
            "limit_price": 2.0,
            "side": "buy",
        },
    )
    assert not v.allowed
    assert "watchlist" in v.reason


def test_option_max_contracts(gates):
    v = gates.evaluate(
        "place_option_order",
        {
            "symbol": "SPY",
            "quantity": 5,
            "limit_price": 2.0,
            "side": "buy",
        },
    )
    assert not v.allowed
    assert "max contracts" in v.reason


def test_option_premium_cap(gates):
    v = gates.evaluate(
        "place_option_order",
        {
            "symbol": "SPY",
            "quantity": 2,
            "limit_price": 5.0,
            "side": "buy",
        },
    )
    assert not v.allowed
    assert "premium" in v.reason


def test_option_long_only_blocks_sell(gates):
    v = gates.evaluate(
        "place_option_order",
        {
            "symbol": "SPY",
            "quantity": 1,
            "limit_price": 2.0,
            "side": "sell",
        },
    )
    assert not v.allowed
    assert "long_only" in v.reason
