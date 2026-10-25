"""Blending prediction files and writing a submission that is checked before upload."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import ID, TARGETS


def average(preds: list[dict[str, np.ndarray]], split: str) -> dict[str, np.ndarray]:
    """Plain mean of several models' probabilities, per target (split = 'oof' or 'test')."""
    if not preds:
        raise ValueError("Need at least one model to blend.")
    return {t: np.mean([p[f"{t}_{split}"] for p in preds], axis=0) for t in TARGETS}


def rank_combos(Y: pd.DataFrame, preds: dict[str, dict[str, np.ndarray]]) -> list[tuple[tuple[str, ...], float]]:
    """OOF mean AUC of every combination of models, best first."""
    names = sorted(preds)
    scores = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            oof = average([preds[k] for k in combo], "oof")
            scores.append((combo, float(np.mean([roc_auc_score(Y[t], oof[t]) for t in TARGETS]))))
    return sorted(scores, key=lambda kv: -kv[1])


def validate(sub: pd.DataFrame, fmt: pd.DataFrame) -> None:
    """Raise ValueError if the submission would be rejected or silently mis-scored."""
    if list(sub.columns) != list(fmt.columns):
        raise ValueError(f"Columns {list(sub.columns)} differ from the format {list(fmt.columns)}.")
    if len(sub) != len(fmt) or not (sub[ID].to_numpy() == fmt[ID].to_numpy()).all():
        raise ValueError("respondent_id values are missing, extra or in a different order.")
    values = sub[list(TARGETS)]
    if values.isna().any().any():
        raise ValueError("Submission contains missing values.")
    if not values.stack().between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")


def build(fmt: pd.DataFrame, test_preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """fmt is indexed by respondent_id (as returned by data.load_format)."""
    sub = fmt.copy()
    for t in TARGETS:
        if len(test_preds[t]) != len(sub):
            raise ValueError(f"{t}: {len(test_preds[t])} predictions for {len(sub)} test rows.")
        sub[t] = test_preds[t]
    sub = sub.reset_index()
    validate(sub, fmt.reset_index())
    return sub
