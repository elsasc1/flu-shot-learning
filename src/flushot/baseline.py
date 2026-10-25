"""Logistic regression baseline (CV ROC AUC 0.8587 on the real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cv import out_of_fold
from .data import TARGETS, add_features, categorical_columns, to_str


def make_pipeline_logreg(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    """Imputation, scaling and one-hot encoding live inside the pipeline, so they are
    re-fitted on the training rows of every fold: no statistics leak from validation."""
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))


def run(X: pd.DataFrame, Y: pd.DataFrame, T: pd.DataFrame) -> dict[str, np.ndarray]:
    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats] + ["n_missing", "behavioral_sum"]
    Xc, Tc = to_str(add_features(X), cats), to_str(add_features(T), cats)
    out: dict[str, np.ndarray] = {}
    for t in TARGETS:
        y = Y[t]

        def fit_predict(tr, va, y=y):
            m = make_pipeline_logreg(nums, cats).fit(Xc.iloc[tr], y.iloc[tr])
            return m.predict_proba(Xc.iloc[va])[:, 1], m.predict_proba(Tc)[:, 1]

        out[f"{t}_oof"], fold_tests = out_of_fold(Xc, y, fit_predict)
        test = np.zeros(len(Tc))
        for p in fold_tests:  # same summation order as the original script
            test += p / 5
        out[f"{t}_test"] = test
    return out
