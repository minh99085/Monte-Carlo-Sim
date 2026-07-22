"""Meta-labeled trading system (v2 rebuild).

Brain = trained secondary model deciding trade/no-trade + size per
TradingView signal (López de Prado meta-labeling). Hand = the existing
Robinhood bot executor (verdict files → mc_bridge → safety gates). Monte
Carlo is a risk overlay only — it never votes on direction.

Honest success criterion: positive deflated Sharpe out-of-sample net of
costs, or a correct refusal to trade. Refusal is the system working.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    import yaml

    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
