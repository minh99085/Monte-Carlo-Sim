#!/usr/bin/env python3
"""Minimal working example: synthetic vision → MC  (small path count) → decision.

Usage (from Monte-Carlo-Sim root)::

    python examples/chart_vision_demo.py
    python examples/chart_vision_demo.py --paths 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_vision_pipeline import run_demo


def main() -> int:
    p = argparse.ArgumentParser(description="Chart vision MC demo")
    p.add_argument("--paths", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--price", type=float, default=190.0)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    decision = run_demo(
        paths=args.paths,
        seed=args.seed,
        ticker=args.ticker,
        price=args.price,
    )
    print(decision.summary_text())
    if args.json_out:
        args.json_out.write_text(
            json.dumps(decision.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
