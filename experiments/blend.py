"""Blend OOF predictions (rank average with grid-searched weights) and a cross-target
stacker (each target's level-2 model sees both targets' level-1 predictions).
Writes submission.csv in the exact submission_format layout."""
import itertools, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

Y = pd.read_csv("data/training_set_labels.csv", index_col="respondent_id")
S = pd.read_csv("data/submission_format.csv", index_col="respondent_id")
TARGETS = ["h1n1_vaccine", "seasonal_vaccine"]
MODELS = ["logreg", "lgbm", "catboost"]
P = {m: np.load(f"preds_{m}.npz") for m in MODELS}
r = lambda a: rankdata(a) / len(a)

# 1) weighted rank average per target
best_w, blend_oof, blend_test = {}, {}, {}
grid = [w for w in itertools.product(np.arange(0, 1.01, 0.05), repeat=3) if abs(sum(w) - 1) < 1e-9]
for t in TARGETS:
    oofs = [r(P[m][f"{t}_oof"]) for m in MODELS]
    scores = [(roc_auc_score(Y[t], sum(w_ * o for w_, o in zip(w, oofs))), w) for w in grid]
    s, w = max(scores)
    best_w[t] = w
    blend_oof[t] = sum(w_ * o for w_, o in zip(w, oofs))
    blend_test[t] = sum(w_ * r(P[m][f"{t}_test"]) for w_, m in zip(w, MODELS))
    print(f"blend {t:17s} AUC={s:.5f} weights={dict(zip(MODELS, np.round(w, 2)))}")
mean_blend = np.mean([roc_auc_score(Y[t], blend_oof[t]) for t in TARGETS])
print(f"blend MEAN AUC={mean_blend:.5f}")

# 2) cross-target stacker (nested CV on OOF features -> honest estimate)
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

F_oof = np.column_stack([logit(P[m][f"{t}_oof"]) for m in MODELS for t in TARGETS])
F_test = np.column_stack([logit(P[m][f"{t}_test"]) for m in MODELS for t in TARGETS])
skf = StratifiedKFold(5, shuffle=True, random_state=7)
stack_oof, stack_test = {}, {}
for t in TARGETS:
    lr = LogisticRegression(C=1.0, max_iter=1000)
    stack_oof[t] = cross_val_predict(lr, F_oof, Y[t], cv=skf, method="predict_proba")[:, 1]
    stack_test[t] = lr.fit(F_oof, Y[t]).predict_proba(F_test)[:, 1]
    print(f"stack {t:17s} AUC={roc_auc_score(Y[t], stack_oof[t]):.5f}")
mean_stack = np.mean([roc_auc_score(Y[t], stack_oof[t]) for t in TARGETS])
print(f"stack MEAN AUC={mean_stack:.5f}")

# pick the better scheme per target; output probabilities in [0,1]
sub = S.copy()
for t in TARGETS:
    a_b = roc_auc_score(Y[t], blend_oof[t]); a_s = roc_auc_score(Y[t], stack_oof[t])
    sub[t] = stack_test[t] if a_s >= a_b else blend_test[t]
    print(f"{t}: using {'stack' if a_s >= a_b else 'blend'}")
final = np.mean([max(roc_auc_score(Y[t], blend_oof[t]), roc_auc_score(Y[t], stack_oof[t])) for t in TARGETS])
print(f"FINAL CV MEAN AUC={final:.5f}")
sub.to_csv("submission.csv")
