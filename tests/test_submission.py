import numpy as np
import pandas as pd
import pytest

from flushot import submission
from flushot.data import ID, TARGETS


@pytest.fixture
def fmt():
    return pd.DataFrame({t: 0.5 for t in TARGETS}, index=pd.Index([10, 11, 12], name=ID))


def preds(a, b):
    return {"h1n1_vaccine": np.array(a, float), "seasonal_vaccine": np.array(b, float)}


def test_build_writes_a_valid_submission(fmt):
    sub = submission.build(fmt, preds([0.1, 0.2, 0.3], [0.9, 0.8, 0.7]))
    assert list(sub.columns) == [ID, *TARGETS]
    assert sub[ID].tolist() == [10, 11, 12]
    assert sub["seasonal_vaccine"].tolist() == [0.9, 0.8, 0.7]


@pytest.mark.parametrize("break_it, message", [
    (lambda s: s.drop(columns="seasonal_vaccine"), "Columns"),
    (lambda s: s.iloc[::-1], "order"),
    (lambda s: s.iloc[:2], "missing, extra"),
    (lambda s: s.assign(h1n1_vaccine=[0.1, np.nan, 0.3]), "missing values"),
    (lambda s: s.assign(h1n1_vaccine=[0.1, 1.2, 0.3]), "between 0 and 1"),
])
def test_validate_rejects_broken_submissions(fmt, break_it, message):
    good = submission.build(fmt, preds([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
    with pytest.raises(ValueError, match=message):
        submission.validate(break_it(good), fmt.reset_index())


def test_build_rejects_wrong_number_of_predictions(fmt):
    with pytest.raises(ValueError, match="2 predictions for 3"):
        submission.build(fmt, preds([0.1, 0.2], [0.3, 0.4]))


def test_average_and_combo_ranking():
    Y = pd.DataFrame({"h1n1_vaccine": [0, 1, 0, 1], "seasonal_vaccine": [0, 0, 1, 1]})
    good = {f"{t}_oof": Y[t].to_numpy() * 0.8 + 0.1 for t in TARGETS}
    bad = {f"{t}_oof": 1 - good[f"{t}_oof"] for t in TARGETS}
    avg = submission.average([good, good], "oof")
    np.testing.assert_allclose(avg["h1n1_vaccine"], good["h1n1_vaccine_oof"])
    ranking = submission.rank_combos(Y, {"good": good, "bad": bad})
    assert ranking[0] == (("good",), 1.0)
    assert ranking[-1] == (("bad",), 0.0)
    with pytest.raises(ValueError):
        submission.average([], "oof")
