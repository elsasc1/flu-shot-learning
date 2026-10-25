"""Loading the competition files and turning them into model inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID = "respondent_id"
TARGETS = ("h1n1_vaccine", "seasonal_vaccine")
FILES = {
    "train": "training_set_features.csv",
    "labels": "training_set_labels.csv",
    "test": "test_set_features.csv",
    "format": "submission_format.csv",
}


def load(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train features, train labels and test features, indexed by respondent_id."""
    d = Path(data_dir)
    missing = [f for f in ("train", "labels", "test") if not (d / FILES[f]).exists()]
    if missing:
        names = ", ".join(FILES[f] for f in missing)
        raise FileNotFoundError(
            f"Missing in {d}: {names}. Download the data from the competition page "
            "(it cannot be redistributed)."
        )
    X = pd.read_csv(d / FILES["train"], index_col=ID)
    Y = pd.read_csv(d / FILES["labels"], index_col=ID)
    T = pd.read_csv(d / FILES["test"], index_col=ID)
    if not X.index.equals(Y.index):
        raise ValueError("Feature and label files do not have the same respondent_id order.")
    return X, Y, T


def load_format(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / FILES["format"], index_col=ID)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Text columns (object dtype, or the string dtype of pandas >= 3)."""
    return [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("str", "string")]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The two engineered features kept after the ablation study."""
    df = df.copy()
    df["n_missing"] = df.isna().sum(axis=1)
    behavioral = [c for c in df.columns if c.startswith("behavioral_")]
    df["behavioral_sum"] = df[behavioral].sum(axis=1)
    return df


def to_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Encode columns as strings so that a missing value becomes its own category.

    Text columns: NaN -> "missing". Float columns: 1.0 -> "1" and NaN -> "nan".
    (Both spellings are kept as in the submitted v2 run, so results are reproduced exactly.)
    """
    df = df.copy()
    for c in cols:
        s = df[c]
        if s.dtype.kind == "f":
            s = s.map(lambda v: "nan" if pd.isna(v) else str(int(v)))
        df[c] = s.astype(object).where(s.notna(), "missing").astype(str)
    return df


def to_category(X: pd.DataFrame, T: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give train and test the same category levels (needed by LightGBM).

    Only the level *names* come from the test set, never any label, so this is not leakage.
    """
    X, T = X.copy(), T.copy()
    for c in cols:
        dtype = pd.CategoricalDtype(sorted(set(X[c]) | set(T[c])))
        X[c], T[c] = X[c].astype(dtype), T[c].astype(dtype)
    return X, T


def prepare(X: pd.DataFrame, T: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the train/test matrices for one model configuration."""
    cats = list(X.columns) if model == "cat_allcat" else categorical_columns(X)
    Xc, Tc = to_str(add_features(X), cats), to_str(add_features(T), cats)
    if model == "lgbm":
        Xc, Tc = to_category(Xc, Tc, cats)
    return Xc, Tc, cats
