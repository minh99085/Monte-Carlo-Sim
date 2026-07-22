"""Secondary model — purged walk-forward logistic regression + isotonic
calibration.

Mirrors the reference fp_modeling.py pattern (StandardScaler +
LogisticRegression) with the JFDS papers' key addition: probability
calibration (isotonic) materially improves fixed sizing. Splits are
purged/embargoed: at each fold boundary, ``embargo_bars`` of signals are
dropped from the end of the train window so a label whose holding window
overlaps the test window can never leak into training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class WalkForwardResult:
    """Out-of-sample calibrated probabilities, one per (kept) signal."""

    indices: List[int] = field(default_factory=list)   # into the signal list
    probs: List[float] = field(default_factory=list)   # calibrated P(label=1)
    fold_of: List[int] = field(default_factory=list)
    n_folds_run: int = 0
    auc: float = float("nan")

    def prob_array(self, n: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i, p in zip(self.indices, self.probs):
            out[i] = p
        return out


def _fit_calibrated(X: np.ndarray, y: np.ndarray):
    """StandardScaler + LogisticRegression wrapped in isotonic calibration.
    Falls back to sigmoid calibration when the train fold is small (isotonic
    needs enough points to be stable)."""
    base = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000))
    method = "isotonic" if y.size >= 100 else "sigmoid"
    clf = CalibratedClassifierCV(base, method=method, cv=3)
    clf.fit(X, y)
    return clf


def purged_walk_forward(
    X: np.ndarray,
    y: np.ndarray,
    order: Sequence[int],
    *,
    n_folds: int = 6,
    embargo: int = 10,
    min_train: int = 60,
) -> WalkForwardResult:
    """Expanding-window walk-forward with an embargo gap.

    ``order`` gives each row's bar index (time order). Rows are assumed
    sorted by time. Fold k trains on all rows whose bar index is at least
    ``embargo`` bars before the test window's first bar, then predicts the
    test window out-of-sample.
    """
    n = X.shape[0]
    res = WalkForwardResult()
    if n < min_train + 10:
        return res
    bounds = np.linspace(min_train, n, n_folds + 1, dtype=int)
    all_true: List[int] = []
    all_prob: List[float] = []
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        test_first_bar = order[lo]
        train_mask = [i for i in range(lo)
                      if order[i] <= test_first_bar - embargo]
        if len(train_mask) < min_train:
            continue
        ytr = y[train_mask]
        if len(np.unique(ytr)) < 2:
            continue
        clf = _fit_calibrated(X[train_mask], ytr)
        p = clf.predict_proba(X[lo:hi])[:, 1]
        res.indices.extend(range(lo, hi))
        res.probs.extend(float(x) for x in p)
        res.fold_of.extend([k] * (hi - lo))
        all_true.extend(int(v) for v in y[lo:hi])
        all_prob.extend(float(x) for x in p)
        res.n_folds_run += 1
    if all_true and len(set(all_true)) == 2:
        from sklearn.metrics import roc_auc_score

        res.auc = float(roc_auc_score(all_true, all_prob))
    return res


def fit_final_model(X: np.ndarray, y: np.ndarray):
    """Model for LIVE decisions: fit on all available (past) labeled signals.
    Only ever called with historical data; live features come later in time
    by construction."""
    return _fit_calibrated(X, y)
