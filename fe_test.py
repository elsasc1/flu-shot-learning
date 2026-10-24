"""Quick A/B: LightGBM 5-fold CV with vs without engineered features."""
import warnings, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
D = "data/"
X = pd.read_csv(D + "training_set_features.csv", index_col="respondent_id")
Y = pd.read_csv(D + "training_set_labels.csv", index_col="respondent_id")
CAT = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("str", "string")]
AGE = {"18 - 34 Years": 0, "35 - 44 Years": 1, "45 - 54 Years": 2, "55 - 64 Years": 3, "65+ Years": 4}
EDU = {"< 12 Years": 0, "12 Years": 1, "Some College": 2, "College Graduate": 3}
INC = {"Below Poverty": 0, "<= $75,000, Above Poverty": 1, "> $75,000": 2}


def base(df):
    df = df.copy()
    df["n_missing"] = df.isna().sum(axis=1)
    df["behavioral_sum"] = df[[c for c in df if c.startswith("behavioral_")]].sum(axis=1)
    return df


def fe(df):
    df = base(df)
    for v, p in [("h1n1", "h1n1"), ("seas", "seasonal")]:
        e, r, s = df[f"opinion_{v}_vacc_effective"], df[f"opinion_{v}_risk"], df[f"opinion_{v}_sick_from_vacc"]
        df[f"{v}_opinion_score"] = e + r - s
        df[f"{v}_doc_x_eff"] = df[f"doctor_recc_{p}"] * e
        df[f"{v}_doc_x_risk"] = df[f"doctor_recc_{p}"] * r
    df["doc_both"] = df["doctor_recc_h1n1"].fillna(-1) * 3 + df["doctor_recc_seasonal"].fillna(-1)
    df["eff_gap"] = df["opinion_h1n1_vacc_effective"] - df["opinion_seas_vacc_effective"]
    df["risk_gap"] = df["opinion_h1n1_risk"] - df["opinion_seas_risk"]
    df["age_num"] = df["age_group"].map(AGE)
    df["edu_num"] = df["education"].map(EDU)
    df["inc_num"] = df["income_poverty"].map(INC)
    df["age_x_chronic"] = df["age_num"] * df["chronic_med_condition"]
    df["insured_x_doc_seas"] = df["health_insurance"].fillna(-1) * 2 + df["doctor_recc_seasonal"].fillna(-1)
    df["ind_occ"] = df["employment_industry"].fillna("na").astype(str) + "_" + df["employment_occupation"].fillna("na").astype(str)
    return df


def run(df, cats):
    df = df.copy()
    for c in cats:
        df[c] = df[c].astype(object).where(df[c].notna(), "missing").astype(str).astype("category")
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    res = []
    for t in Y.columns:
        oof = np.zeros(len(df))
        for tr, va in skf.split(df, Y[t]):
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.01, num_leaves=24, min_child_samples=40,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3,
                                   cat_smooth=20, random_state=42, verbose=-1, n_jobs=2)
            m.fit(df.iloc[tr], Y[t].iloc[tr], eval_set=[(df.iloc[va], Y[t].iloc[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(300, verbose=False)])
            oof[va] = m.predict_proba(df.iloc[va])[:, 1]
        res.append(roc_auc_score(Y[t], oof))
    return res


b = run(base(X), CAT)
print(f"BASE h1n1={b[0]:.5f} seas={b[1]:.5f} mean={np.mean(b):.5f}", flush=True)
f = run(fe(X), CAT + ["ind_occ"])
print(f"FE   h1n1={f[0]:.5f} seas={f[1]:.5f} mean={np.mean(f):.5f}  delta={np.mean(f) - np.mean(b):+.5f}", flush=True)
