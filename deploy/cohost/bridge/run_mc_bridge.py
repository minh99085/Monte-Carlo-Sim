#!/usr/bin/env python3
"""Run the Monte-Carlo-Sim → Robinhood paper bridge in a poll loop.

Watches the mounted Monte-Carlo-Sim output directories for new verdict JSON
files, maps fresh TRADE verdicts through the safety gates, and appends every
outcome to ``$RH_DATA_DIR/mc_bridge_ledger.jsonl``. Phase 1 makes **no**
Robinhood API calls — see ``engine/robinhood/mc_bridge.py``.

Environment:
    MC_VERDICTS_DIRS   colon-separated verdict dirs
                       (default "/mc-outputs/verdicts:/mc-outputs/paper_verdicts")
    MC_BRIDGE_POLL_S   seconds between passes (default 300)
    MC_BRIDGE_MAX_AGE_H  max verdict age in hours (default 48)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.robinhood.audit_log import AuditLog
from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.mc_bridge import DEFAULT_MAX_AGE_HOURS, process_once
from engine.robinhood.safety_gates import RobinhoodSafetyGates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hermes.robinhood.mc_bridge")


def main() -> int:
    config = RobinhoodConfig.from_env()
    dirs = [
        Path(p) for p in os.getenv(
            "MC_VERDICTS_DIRS",
            "/mc-outputs/verdicts:/mc-outputs/paper_verdicts",
        ).split(":") if p
    ]
    poll_s = float(os.getenv("MC_BRIDGE_POLL_S", "300"))
    max_age_h = float(os.getenv("MC_BRIDGE_MAX_AGE_H", str(DEFAULT_MAX_AGE_HOURS)))

    audit = AuditLog(config.data_dir)
    gates = RobinhoodSafetyGates(config, audit)

    logger.info("MC bridge starting (paper mode, no Robinhood calls): dirs=%s poll=%.0fs max_age=%.0fh",
                [str(d) for d in dirs], poll_s, max_age_h)
    missing = [d for d in dirs if not d.is_dir()]
    if missing:
        logger.warning("verdict dir(s) not found yet (will keep checking): %s",
                       [str(d) for d in missing])

    while True:
        try:
            summary = process_once(dirs, config, gates=gates, audit=audit,
                                   max_age_hours=max_age_h)
            if summary["new"]:
                logger.info("pass: %s", summary)
        except Exception:  # noqa: BLE001 — keep the loop alive
            logger.exception("bridge pass failed; retrying next poll")
        time.sleep(poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
