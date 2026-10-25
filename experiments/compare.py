"""Compare OOF AUC of blend combinations; optionally write submission from a chosen combo.
Usage: python compare.py [write combo_name]"""
import sys, itertools, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

Y = pd.read_csv("data/training_set_labels.csv", index_col="respondent_id")
S = pd.read_csv("data/submission_format.csv", index_col="respondent_id")
FILES = {"lgbm2": "preds2_lgbm.npz", "allcat2": "preds2_cat_allcat.npz",
         "catnum2": "preds2_cat_num.npz", "cat1": "preds_catboost.npz"}
P = {}
for k, f in FILES.items():
    try:
        P[k] = np.load(f)
    except FileNotFoundError:
        pass
names = list(P)
results = {}
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        sc = [roc_auc_score(Y[t], np.mean([P[k][f"{t}_oof"] for k in combo], axis=0)) for t in Y.columns]
        results[combo] = sc
for combo, sc in sorted(results.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"{'+'.join(combo):30s} h1n1={sc[0]:.5f} seas={sc[1]:.5f} mean={np.mean(sc):.5f}")

if len(sys.argv) > 2 and sys.argv[1] == "write":
    combo = tuple(sys.argv[2].split("+"))
    sub = S.copy()
    for t in Y.columns:
        sub[t] = np.mean([P[k][f"{t}_test"] for k in combo], axis=0)
    sub.to_csv("submission_v2.csv")
    chk, fmt = pd.read_csv("submission_v2.csv"), pd.read_csv("data/submission_format.csv")
    assert list(chk.columns) == list(fmt.columns)
    assert (chk.respondent_id.values == fmt.respondent_id.values).all()
    assert chk.isna().sum().sum() == 0
    assert chk[list(Y.columns)].stack().between(0, 1).all()
    v1 = pd.read_csv("submission.csv")
    print("written", combo, chk.shape, "checks OK; corr with v1:",
          {t: round(np.corrcoef(chk[t], v1[t])[0, 1], 4) for t in Y.columns})
