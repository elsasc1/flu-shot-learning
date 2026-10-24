"""v2: 5-fold CV (OOF + best iterations), then full-data refits averaged over seeds for test.
Usage: python train2.py <config>   configs: lgbm | cat_num | cat_allcat
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
D = "data/"
X = pd.read_csv(D + "training_set_features.csv", index_col="respondent_id")
Y = pd.read_csv(D + "training_set_labels.csv", index_col="respondent_id")
T = pd.read_csv(D + "test_set_features.csv", index_col="respondent_id")
TARGETS = ["h1n1_vaccine", "seasonal_vaccine"]
CAT0 = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("str", "string")]


def add_features(df):
    df = df.copy()
    df["n_missing"] = df.isna().sum(axis=1)
    beh = [c for c in df.columns if c.startswith("behavioral_")]
    df["behavioral_sum"] = df[beh].sum(axis=1)
    return df


def to_str(df, cols):
    df = df.copy()
    for c in cols:
        s = df[c]
        if s.dtype.kind == "f":
            s = s.map(lambda v: "nan" if pd.isna(v) else str(int(v)))
        df[c] = s.astype(object).where(s.notna(), "missing").astype(str)
    return df


cfg = sys.argv[1]
Xf, Tf = add_features(X), add_features(T)
if cfg == "cat_allcat":
    CAT = [c for c in X.columns]           # every raw survey answer as a category (NaN = own level)
else:
    CAT = CAT0
Xc, Tc = to_str(Xf, CAT), to_str(Tf, CAT)
if cfg == "lgbm":
    for c in CAT:
        dt = pd.CategoricalDtype(sorted(set(Xc[c]) | set(Tc[c])))
        Xc[c], Tc[c] = Xc[c].astype(dt), Tc[c].astype(dt)

SEEDS_FULL = {"lgbm": [0, 1, 2, 3, 4], "cat_num": [0, 1, 2], "cat_allcat": [0, 1, 2]}[cfg]


def make(seed, n_iter=None):
    if cfg == "lgbm":
        return lgb.LGBMClassifier(n_estimators=n_iter or 5000, learning_rate=0.01, num_leaves=24,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.5, reg_lambda=3, cat_smooth=20,
                                  random_state=seed, verbose=-1, n_jobs=2)
    depth = 6 if cfg == "cat_num" else 5
    return CatBoostClassifier(iterations=n_iter or 4000, learning_rate=0.03, depth=depth,
                              l2_leaf_reg=5, eval_metric="AUC", random_seed=seed, verbose=0,
                              thread_count=2, **({} if n_iter else {"early_stopping_rounds": 200}))


def fit(m, Xtr, ytr, Xva=None, yva=None):
    if cfg == "lgbm":
        kw = {}
        if Xva is not None:
            kw = dict(eval_set=[(Xva, yva)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(300, verbose=False)])
        m.fit(Xtr, ytr, **kw)
        return m.best_iteration_ if Xva is not None else None
    kw = dict(cat_features=CAT)
    if Xva is not None:
        kw.update(eval_set=(Xva, yva), use_best_model=True)
    m.fit(Xtr, ytr, **kw)
    return m.get_best_iteration() if Xva is not None else None


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
out = {}
for t in TARGETS:
    y = Y[t]
    oof, iters = np.zeros(len(Xc)), []
    for tr, va in skf.split(Xc, y):
        m = make(42)
        iters.append(fit(m, Xc.iloc[tr], y.iloc[tr], Xc.iloc[va], y.iloc[va]))
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
    n_full = int(np.mean(iters) * 1.15)
    print(f"{cfg:10s} {t:17s} CV AUC={roc_auc_score(y, oof):.5f} iters={iters} -> full {n_full}", flush=True)
    test = np.zeros(len(Tc))
    for s in SEEDS_FULL:
        m = make(s, n_full)
        fit(m, Xc, y)
        test += m.predict_proba(Tc)[:, 1] / len(SEEDS_FULL)
    out[f"{t}_oof"], out[f"{t}_test"] = oof, test
np.savez(f"preds2_{cfg}.npz", **out)
print(f"{cfg:10s} MEAN CV AUC={np.mean([roc_auc_score(Y[t], out[f'{t}_oof']) for t in TARGETS]):.5f}", flush=True)
