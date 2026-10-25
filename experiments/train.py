"""Flu Shot Learning (DrivenData) — CV pipeline.
Metric: mean ROC AUC over h1n1_vaccine and seasonal_vaccine.
Models: logistic regression baseline, CatBoost, LightGBM; OOF-weighted blend.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
D = "data/"
X = pd.read_csv(D + "training_set_features.csv", index_col="respondent_id")
Y = pd.read_csv(D + "training_set_labels.csv", index_col="respondent_id")
T = pd.read_csv(D + "test_set_features.csv", index_col="respondent_id")
TARGETS = ["h1n1_vaccine", "seasonal_vaccine"]
CAT = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("str", "string")]
NUM = [c for c in X.columns if c not in CAT]


def prep_cat(df):
    df = df.copy()
    for c in CAT:
        df[c] = df[c].astype(object).where(df[c].notna(), "missing").astype(str)
    return df


def add_features(df):
    df = df.copy()
    df["n_missing"] = df.isna().sum(axis=1)
    beh = [c for c in df.columns if c.startswith("behavioral_")]
    df["behavioral_sum"] = df[beh].sum(axis=1)
    return df


Xf, Tf = add_features(X), add_features(T)
NUMF = NUM + ["n_missing", "behavioral_sum"]
Xc, Tc = prep_cat(Xf), prep_cat(Tf)

SEED = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def run_logreg(y):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMF),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    oof, test = np.zeros(len(Xc)), np.zeros(len(Tc))
    for tr, va in skf.split(Xc, y):
        m = make_pipeline(pre, LogisticRegression(C=0.1, max_iter=2000))
        m.fit(Xc.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        test += m.predict_proba(Tc)[:, 1] / 5
    return oof, test


def run_catboost(y):
    oof, test = np.zeros(len(Xc)), np.zeros(len(Tc))
    for tr, va in skf.split(Xc, y):
        m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                               eval_metric="AUC", random_seed=SEED, verbose=0,
                               early_stopping_rounds=200, thread_count=2)
        m.fit(Xc.iloc[tr], y.iloc[tr], cat_features=CAT,
              eval_set=(Xc.iloc[va], y.iloc[va]), use_best_model=True)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
        test += m.predict_proba(Tc)[:, 1] / 5
    return oof, test


def run_lgbm(y):
    Xl, Tl = Xc.copy(), Tc.copy()
    for c in CAT:
        cats = pd.CategoricalDtype(sorted(set(Xl[c]) | set(Tl[c])))
        Xl[c], Tl[c] = Xl[c].astype(cats), Tl[c].astype(cats)
    oof, test = np.zeros(len(Xl)), np.zeros(len(Tl))
    for tr, va in skf.split(Xl, y):
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.01, num_leaves=24,
                               min_child_samples=40, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, reg_lambda=3, cat_smooth=20,
                               random_state=SEED, verbose=-1, n_jobs=2)
        m.fit(Xl.iloc[tr], y.iloc[tr], eval_set=[(Xl.iloc[va], y.iloc[va])],
              eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va] = m.predict_proba(Xl.iloc[va])[:, 1]
        test += m.predict_proba(Tl)[:, 1] / 5
    return oof, test


RUNNERS = {"logreg": run_logreg, "catboost": run_catboost, "lgbm": run_lgbm}
which = sys.argv[1:] or list(RUNNERS)
for name in which:
    res = {}
    for t in TARGETS:
        oof, test = RUNNERS[name](Y[t])
        res[t] = (oof, test)
        print(f"{name:9s} {t:17s} AUC={roc_auc_score(Y[t], oof):.5f}", flush=True)
    np.savez(f"preds_{name}.npz", **{f"{t}_oof": res[t][0] for t in TARGETS},
             **{f"{t}_test": res[t][1] for t in TARGETS})
    mean = np.mean([roc_auc_score(Y[t], res[t][0]) for t in TARGETS])
    print(f"{name:9s} MEAN AUC={mean:.5f}", flush=True)
