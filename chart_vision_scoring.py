"""
Chart2CSV-style evaluation scoring for chart vision extractions.

Expected evaluation set layout::

    eval_set/
      case_001/
        image.png          # (optional for offline scoring)
        ground_truth.json  # expert JSON matching ChartExtractionResult fields
        prediction.json    # model output (same schema)

Or a single JSONL where each line has ``{"id", "truth": {...}, "pred": {...}}``.

Usage outline::

    python chart_vision_scoring.py --dir eval_set/
    python chart_vision_scoring.py --jsonl predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chart_vision_models import ChartExtractionResult, LevelKind


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_ticker(truth: ChartExtractionResult, pred: ChartExtractionResult) -> float:
    return 1.0 if truth.ticker.upper() == pred.ticker.upper() else 0.0


def score_bias(truth: ChartExtractionResult, pred: ChartExtractionResult) -> float:
    return 1.0 if truth.bias == pred.bias else 0.0


def score_rsi_abs_error(
    truth: ChartExtractionResult, pred: ChartExtractionResult
) -> Optional[float]:
    t = truth.indicators.rsi.value
    p = pred.indicators.rsi.value
    if t is None or p is None:
        return None
    return abs(float(t) - float(p))


def _level_prices(levels: List[Any], kind: Optional[LevelKind] = None) -> List[float]:
    out: List[float] = []
    for lv in levels:
        k = lv.kind if hasattr(lv, "kind") else lv.get("kind")
        price = lv.price if hasattr(lv, "price") else lv.get("price")
        if price is None:
            continue
        if kind is not None and str(k) != kind.value and k != kind:
            continue
        out.append(float(price))
    return out


def score_levels(
    truth: ChartExtractionResult,
    pred: ChartExtractionResult,
    *,
    rel_tol: float = 0.01,
) -> Dict[str, float]:
    """
    Match predicted levels to ground-truth levels within relative tolerance.

    Returns precision, recall, F1 over union of support+resistance prices.
    """
    t_prices = _level_prices(truth.levels)
    p_prices = _level_prices(pred.levels)
    if not t_prices and not p_prices:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_truth": 0, "n_pred": 0}
    if not t_prices:
        return {
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "n_truth": 0,
            "n_pred": float(len(p_prices)),
        }
    if not p_prices:
        return {
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_truth": float(len(t_prices)),
            "n_pred": 0,
        }

    matched_t = set()
    matched_p = set()
    for i, tp in enumerate(t_prices):
        for j, pp in enumerate(p_prices):
            if j in matched_p:
                continue
            if abs(tp - pp) / max(abs(tp), 1e-9) <= rel_tol:
                matched_t.add(i)
                matched_p.add(j)
                break

    tp_count = len(matched_t)
    precision = tp_count / len(p_prices)
    recall = tp_count / len(t_prices)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_truth": float(len(t_prices)),
        "n_pred": float(len(p_prices)),
    }


def score_pair(
    truth: ChartExtractionResult,
    pred: ChartExtractionResult,
    *,
    level_rel_tol: float = 0.01,
) -> Dict[str, Any]:
    rsi_err = score_rsi_abs_error(truth, pred)
    levels = score_levels(truth, pred, rel_tol=level_rel_tol)
    return {
        "ticker_accuracy": score_ticker(truth, pred),
        "bias_accuracy": score_bias(truth, pred),
        "rsi_abs_error": rsi_err,
        "level_precision": levels["precision"],
        "level_recall": levels["recall"],
        "level_f1": levels["f1"],
        "timeframe_match": 1.0
        if truth.timeframe.lower() == pred.timeframe.lower()
        else 0.0,
        "overall_confidence_pred": pred.confidence.overall,
    }


def _load_extraction(data: Dict[str, Any]) -> ChartExtractionResult:
    return ChartExtractionResult.model_validate(data)


def iter_cases_from_dir(root: Path) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        truth_p = case_dir / "ground_truth.json"
        pred_p = case_dir / "prediction.json"
        if not truth_p.is_file() or not pred_p.is_file():
            continue
        truth = json.loads(truth_p.read_text(encoding="utf-8"))
        pred = json.loads(pred_p.read_text(encoding="utf-8"))
        yield case_dir.name, truth, pred


def iter_cases_from_jsonl(
    path: Path,
) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = str(row.get("id") or i)
            yield cid, row["truth"], row["pred"]


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {"n": 0}

    def mean(key: str) -> Optional[float]:
        vals = [s[key] for s in scores if s.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return {
        "n": len(scores),
        "ticker_accuracy": mean("ticker_accuracy"),
        "bias_accuracy": mean("bias_accuracy"),
        "mean_rsi_abs_error": mean("rsi_abs_error"),
        "mean_level_f1": mean("level_f1"),
        "mean_level_precision": mean("level_precision"),
        "mean_level_recall": mean("level_recall"),
        "timeframe_accuracy": mean("timeframe_match"),
        "mean_pred_confidence": mean("overall_confidence_pred"),
    }


def run_scoring(
    *,
    eval_dir: Optional[Path] = None,
    jsonl: Optional[Path] = None,
    level_rel_tol: float = 0.01,
) -> Dict[str, Any]:
    cases: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    if eval_dir is not None:
        cases.extend(iter_cases_from_dir(eval_dir))
    if jsonl is not None:
        cases.extend(iter_cases_from_jsonl(jsonl))

    per_case: List[Dict[str, Any]] = []
    for cid, truth_raw, pred_raw in cases:
        truth = _load_extraction(truth_raw)
        pred = _load_extraction(pred_raw)
        s = score_pair(truth, pred, level_rel_tol=level_rel_tol)
        s["id"] = cid
        per_case.append(s)

    return {"aggregate": aggregate(per_case), "cases": per_case}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score chart vision extractions")
    p.add_argument("--dir", type=Path, default=None, help="Eval set directory")
    p.add_argument("--jsonl", type=Path, default=None, help="JSONL of truth/pred pairs")
    p.add_argument("--level-tol", type=float, default=0.01, help="Relative level match tol")
    p.add_argument("--out", type=Path, default=None, help="Write full JSON report")
    args = p.parse_args(argv)

    if args.dir is None and args.jsonl is None:
        p.error("Provide --dir and/or --jsonl")

    report = run_scoring(
        eval_dir=args.dir,
        jsonl=args.jsonl,
        level_rel_tol=args.level_tol,
    )
    print(json.dumps(report["aggregate"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
