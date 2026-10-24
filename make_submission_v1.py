"""Soumission v1 : moyenne 50/50 des probabilités LightGBM + CatBoost (sorties de train.py),
puis vérifications de format avant envoi."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

Y = pd.read_csv("data/training_set_labels.csv", index_col="respondent_id")
S = pd.read_csv("data/submission_format.csv", index_col="respondent_id")
P = {m: np.load(f"preds_{m}.npz") for m in ["lgbm", "catboost"]}

sub, scores = S.copy(), []
for t in S.columns:
    oof = 0.5 * P["lgbm"][f"{t}_oof"] + 0.5 * P["catboost"][f"{t}_oof"]
    scores.append(roc_auc_score(Y[t], oof)); print(t, round(scores[-1], 5))
    sub[t] = 0.5 * P["lgbm"][f"{t}_test"] + 0.5 * P["catboost"][f"{t}_test"]
print("MEAN", round(np.mean(scores), 5))
sub.to_csv("submission.csv")

# Vérifications : mêmes colonnes, mêmes identifiants dans le même ordre, pas de NaN, probas dans [0, 1]
chk = pd.read_csv("submission.csv")
fmt = pd.read_csv("data/submission_format.csv")
assert list(chk.columns) == list(fmt.columns)
assert (chk.respondent_id.values == fmt.respondent_id.values).all()
assert chk.isna().sum().sum() == 0
assert chk[["h1n1_vaccine", "seasonal_vaccine"]].stack().between(0, 1).all()
print(chk.shape, "checks OK")
