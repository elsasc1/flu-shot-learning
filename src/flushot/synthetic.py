"""Fake data with the exact schema of the competition files.

The real data cannot be redistributed, so tests and the CI smoke run use this instead.
The labels depend on a few features (doctor recommendation, opinions), so models can
learn something and AUC-based tests are meaningful.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FILES, ID, TARGETS

BINARY = [
    "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
    "behavioral_wash_hands", "behavioral_large_gatherings", "behavioral_outside_home",
    "behavioral_touch_face", "doctor_recc_h1n1", "doctor_recc_seasonal",
    "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance",
]
ORDINAL = {
    "h1n1_concern": 4, "h1n1_knowledge": 3,
    "opinion_h1n1_vacc_effective": 5, "opinion_h1n1_risk": 5, "opinion_h1n1_sick_from_vacc": 5,
    "opinion_seas_vacc_effective": 5, "opinion_seas_risk": 5, "opinion_seas_sick_from_vacc": 5,
    "household_adults": 4, "household_children": 4,
}
TEXT = {
    "age_group": ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"],
    "education": ["< 12 Years", "12 Years", "Some College", "College Graduate"],
    "race": ["White", "Black", "Hispanic", "Other or Multiple"],
    "sex": ["Female", "Male"],
    "income_poverty": ["Below Poverty", "<= $75,000, Above Poverty", "> $75,000"],
    "marital_status": ["Married", "Not Married"],
    "rent_or_own": ["Own", "Rent"],
    "employment_status": ["Employed", "Not in Labor Force", "Unemployed"],
    "hhs_geo_region": ["lzgpxyit", "fpwskwrf", "qufhixun", "oxchjgsf", "kbazzjca"],
    "census_msa": ["MSA, Not Principle  City", "MSA, Principle City", "Non-MSA"],
    "employment_industry": ["fcxhlnwr", "wxleyezf", "ldnlellj", "pxcmvdjn", "atmlpfrs"],
    "employment_occupation": ["xtkaffoo", "mxkfnird", "emcorrxb", "cmhcxjea", "xgwztkwe"],
}
COLUMNS = [
    "h1n1_concern", "h1n1_knowledge", *BINARY[:7], "doctor_recc_h1n1", "doctor_recc_seasonal",
    "chronic_med_condition", "child_under_6_months", "health_worker", "health_insurance",
    "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc",
    "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
    "age_group", "education", "race", "sex", "income_poverty", "marital_status", "rent_or_own",
    "employment_status", "hhs_geo_region", "census_msa", "household_adults", "household_children",
    "employment_industry", "employment_occupation",
]


def make(n_train: int = 600, n_test: int = 200, seed: int = 0):
    """Return (features, labels, test features, submission format), like the real files."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    cols = {}
    for c in BINARY:
        cols[c] = rng.integers(0, 2, n).astype(float)
    for c, k in ORDINAL.items():
        cols[c] = rng.integers(0 if c.startswith(("h1n1_", "household")) else 1, k, n).astype(float)
    for c, levels in TEXT.items():
        cols[c] = rng.choice(levels, n).astype(object)
    df = pd.DataFrame(cols)[COLUMNS]
    for c in COLUMNS:  # missing values, much more frequent for employment fields as in the real data
        rate = 0.5 if c in ("employment_industry", "employment_occupation", "health_insurance") else 0.05
        df.loc[rng.random(n) < rate, c] = np.nan
    logit_h1n1 = (2.0 * df["doctor_recc_h1n1"].fillna(0) + 0.6 * df["opinion_h1n1_vacc_effective"].fillna(3)
                  - 3.0 + rng.normal(0, 1, n))
    logit_seas = (1.5 * df["doctor_recc_seasonal"].fillna(0) + 0.5 * df["opinion_seas_vacc_effective"].fillna(3)
                  + 1.5 * (logit_h1n1 > 0) - 2.5 + rng.normal(0, 1, n))
    labels = pd.DataFrame({TARGETS[0]: (logit_h1n1 > 0).astype(int), TARGETS[1]: (logit_seas > 0).astype(int)})
    df.index = labels.index = pd.RangeIndex(n, name=ID)
    X, Y, T = df.iloc[:n_train], labels.iloc[:n_train], df.iloc[n_train:]
    fmt = pd.DataFrame({t: 0.5 for t in TARGETS}, index=T.index)
    return X, Y, T, fmt


def write(data_dir: str | Path, **kwargs) -> Path:
    """Write the four fake CSV files, with the competition file names, into data_dir."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    X, Y, T, fmt = make(**kwargs)
    X.to_csv(d / FILES["train"])
    Y.to_csv(d / FILES["labels"])
    T.to_csv(d / FILES["test"])
    fmt.to_csv(d / FILES["format"])
    return d
