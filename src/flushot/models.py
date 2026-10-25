"""Model configurations of the v2 submission."""

from __future__ import annotations

import warnings

MODELS = ("lgbm", "cat_num", "cat_allcat")
FULL_FIT_SEEDS = {"lgbm": [0, 1, 2, 3, 4], "cat_num": [0, 1, 2], "cat_allcat": [0, 1, 2]}
CV_SEED = 42
REFIT_FACTOR = 1.15  # full-data refit uses 15% more trees: it sees 25% more data than a fold model


def check_name(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODELS)}.")


def make_model(model: str, seed: int, n_iter: int | None = None):
    """n_iter=None: many trees + early stopping (CV). n_iter set: fixed size (full-data refit)."""
    check_name(model)
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=n_iter or 5000, learning_rate=0.01, num_leaves=24,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            reg_lambda=3, cat_smooth=20, random_state=seed, verbose=-1, n_jobs=2,
        )
    from catboost import CatBoostClassifier

    early = {} if n_iter else {"early_stopping_rounds": 200}
    return CatBoostClassifier(
        iterations=n_iter or 4000, learning_rate=0.03, depth=6 if model == "cat_num" else 5,
        l2_leaf_reg=5, eval_metric="AUC", random_seed=seed, verbose=0, thread_count=2, **early,
    )


def fit_model(model: str, m, X_tr, y_tr, cats: list[str], X_va=None, y_va=None) -> int | None:
    """Fit; with a validation set, use early stopping and return the best iteration."""
    with_valid = X_va is not None
    if model == "lgbm":
        import lightgbm as lgb

        kw = {}
        if with_valid:
            kw = dict(eval_set=[(X_va, y_va)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        with warnings.catch_warnings():
            # LightGBM >= 4.7 renames eval_set; keeping it supports older versions too.
            warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated")
            m.fit(X_tr, y_tr, **kw)
        return m.best_iteration_ if with_valid else None
    kw = dict(cat_features=cats)
    if with_valid:
        kw.update(eval_set=(X_va, y_va), use_best_model=True)
    m.fit(X_tr, y_tr, **kw)
    return m.get_best_iteration() if with_valid else None
