import numpy as np
import pandas as pd
import pytest

from flushot import data


def test_add_features_counts_missing_and_behaviours():
    df = pd.DataFrame({"behavioral_a": [1.0, 0.0, np.nan], "behavioral_b": [1.0, 1.0, 0.0], "age": ["x", None, "y"]})
    out = data.add_features(df)
    assert out["n_missing"].tolist() == [0, 1, 1]
    assert out["behavioral_sum"].tolist() == [2.0, 1.0, 0.0]
    assert "n_missing" not in df.columns  # input is not modified


def test_to_str_keeps_missing_as_its_own_category():
    df = pd.DataFrame({"text": ["a", None], "num": [1.0, np.nan]})
    out = data.to_str(df, ["text", "num"])
    assert out["text"].tolist() == ["a", "missing"]
    assert out["num"].tolist() == ["1", "nan"]


def test_categorical_columns_are_the_text_columns(fake):
    X = fake[0]
    cats = data.categorical_columns(X)
    assert len(cats) == 12
    assert all(X[c].dtype.kind != "f" for c in cats)


@pytest.mark.parametrize("model", ["lgbm", "cat_num", "cat_allcat"])
def test_prepare_gives_train_and_test_the_same_columns(fake, model):
    X, _, T, _ = fake
    Xc, Tc, cats = data.prepare(X, T, model)
    assert list(Xc.columns) == list(Tc.columns) == list(X.columns) + ["n_missing", "behavioral_sum"]
    assert len(cats) == (35 if model == "cat_allcat" else 12)
    for c in cats:
        assert not Xc[c].isna().any()
        if model == "lgbm":  # LightGBM needs identical category levels in train and test
            assert list(Xc[c].cat.categories) == list(Tc[c].cat.categories)


def test_load_explains_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot be redistributed"):
        data.load(tmp_path)


def test_load_reads_the_competition_files(data_dir):
    X, Y, T = data.load(data_dir)
    assert X.shape == (400, 35) and T.shape == (120, 35)
    assert list(Y.columns) == list(data.TARGETS)
    assert X.index.name == data.ID
