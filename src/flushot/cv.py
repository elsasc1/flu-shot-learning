"""Cross-validation, full-data refit and seed averaging."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .data import TARGETS, prepare
from .models import CV_SEED, FULL_FIT_SEEDS, REFIT_FACTOR, fit_model, make_model


def folds(y: pd.Series, n_splits: int = 5, seed: int = CV_SEED) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds: same positive rate in every fold, same split for a given seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def out_of_fold(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, object]],
    n_splits: int = 5,
    seed: int = CV_SEED,
) -> tuple[np.ndarray, list]:
    """Generic OOF loop. fit_predict(train_idx, valid_idx) trains on train_idx only and
    returns (predictions for valid_idx, extra info). Each row is predicted exactly once,
    by a model that never saw it."""
    oof = np.full(len(X), np.nan)
    infos = []
    for tr, va in folds(y, n_splits, seed):
        pred, info = fit_predict(tr, va)
        oof[va] = pred
        infos.append(info)
    if np.isnan(oof).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")
    return oof, infos


def run(model: str, X: pd.DataFrame, Y: pd.DataFrame, T: pd.DataFrame,
        seeds: list[int] | None = None, log: Callable[[str], None] = print) -> dict[str, np.ndarray]:
    """For each target: 5-fold CV (OOF + best iterations), then full-data refits averaged
    over seeds for the test predictions. Returns {target_oof, target_test} arrays."""
    Xc, Tc, cats = prepare(X, T, model)
    seeds = FULL_FIT_SEEDS[model] if seeds is None else seeds
    out: dict[str, np.ndarray] = {}
    for t in TARGETS:
        y = Y[t]

        def fit_predict(tr, va, y=y):
            m = make_model(model, CV_SEED)
            best = fit_model(model, m, Xc.iloc[tr], y.iloc[tr], cats, Xc.iloc[va], y.iloc[va])
            return m.predict_proba(Xc.iloc[va])[:, 1], best

        oof, iters = out_of_fold(Xc, y, fit_predict)
        n_full = int(np.mean(iters) * REFIT_FACTOR)
        log(f"{model:10s} {t:17s} CV AUC={roc_auc_score(y, oof):.5f} iters={iters} -> full {n_full}")
        test = np.zeros(len(Tc))
        for s in seeds:
            m = make_model(model, s, n_full)
            fit_model(model, m, Xc, y, cats)
            test += m.predict_proba(Tc)[:, 1] / len(seeds)
        out[f"{t}_oof"], out[f"{t}_test"] = oof, test
    log(f"{model:10s} MEAN CV AUC={mean_auc(Y, out):.5f}")
    return out


def mean_auc(Y: pd.DataFrame, preds: dict[str, np.ndarray]) -> float:
    """Competition metric on OOF predictions: ROC AUC averaged over the two targets."""
    return float(np.mean([roc_auc_score(Y[t], preds[f"{t}_oof"]) for t in TARGETS]))
