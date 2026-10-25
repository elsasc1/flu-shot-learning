import numpy as np
from sklearn.metrics import roc_auc_score

from flushot import baseline, cv, data


def test_folds_partition_the_rows_and_keep_class_balance(fake):
    y = fake[1]["h1n1_vaccine"]
    seen = []
    for tr, va in cv.folds(y):
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == len(y)
        assert abs(y.iloc[va].mean() - y.mean()) < 0.05
        seen.extend(va)
    assert sorted(seen) == list(range(len(y)))  # every row validated exactly once


def test_folds_are_deterministic(fake):
    y = fake[1]["seasonal_vaccine"]
    a = [va.tolist() for _, va in cv.folds(y, seed=42)]
    b = [va.tolist() for _, va in cv.folds(y, seed=42)]
    c = [va.tolist() for _, va in cv.folds(y, seed=7)]
    assert a == b and a != c


def test_out_of_fold_never_predicts_a_row_the_model_was_trained_on(fake):
    X, Y = fake[0], fake[1]
    trained_on = []

    def fit_predict(tr, va):
        trained_on.append(set(tr))
        assert not set(va) & set(tr)
        return np.full(len(va), len(trained_on), dtype=float), None

    oof, _ = cv.out_of_fold(X, Y["h1n1_vaccine"], fit_predict)
    assert not np.isnan(oof).any()
    for fold_id, rows in enumerate(trained_on, start=1):
        predicted_here = set(np.where(oof == fold_id)[0])
        assert not predicted_here & rows


def test_baseline_preprocessing_is_fitted_on_training_rows_only(fake):
    X, Y, T, _ = fake
    Xc = data.to_str(data.add_features(X), data.categorical_columns(X))
    nums = ["h1n1_concern", "n_missing"]
    tr = np.arange(200)
    pipe = baseline.make_pipeline_logreg(nums, []).fit(Xc[nums].iloc[tr], Y["h1n1_vaccine"].iloc[tr])
    imputer = pipe[0].named_transformers_["num"][0]
    assert np.allclose(imputer.statistics_, Xc[nums].iloc[tr].median())


def test_lgbm_run_is_reproducible_and_learns(fake):
    X, Y, T, _ = fake
    a = cv.run("lgbm", X, Y, T, seeds=[0, 1], log=lambda _: None)
    b = cv.run("lgbm", X, Y, T, seeds=[0, 1], log=lambda _: None)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert set(a) == {f"{t}_{s}" for t in data.TARGETS for s in ("oof", "test")}
    assert len(a["h1n1_vaccine_test"]) == len(T)
    assert ((a["h1n1_vaccine_test"] >= 0) & (a["h1n1_vaccine_test"] <= 1)).all()
    # the synthetic labels depend on the doctor's recommendation, so the model must beat chance
    assert roc_auc_score(Y["h1n1_vaccine"], a["h1n1_vaccine_oof"]) > 0.75
