"""End-to-end smoke test: fake data -> train -> compare -> submit, through the command line."""

import pandas as pd
import pytest

from flushot import cli, data, submission


def test_full_pipeline_from_the_command_line(tmp_path, capsys):
    d, out = tmp_path / "data", tmp_path / "outputs"
    common = ["--data-dir", str(d), "--out-dir", str(out)]
    cli.main(["--data-dir", str(d), "fake-data", "--n-train", "300", "--n-test", "80"])
    cli.main([*common, "train", "lgbm"])
    cli.main([*common, "baseline"])
    cli.main([*common, "compare"])
    cli.main([*common, "submit", "lgbm", "logreg", "--output", str(tmp_path / "sub.csv")])
    printed = capsys.readouterr().out
    assert "lgbm+logreg" in printed and "Checks OK" in printed
    sub = pd.read_csv(tmp_path / "sub.csv")
    submission.validate(sub, data.load_format(d).reset_index())
    assert len(sub) == 80


def test_submit_without_trained_model_says_what_to_run(data_dir, tmp_path):
    with pytest.raises(SystemExit, match="python -m flushot train lgbm"):
        cli.main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "o"), "submit", "lgbm"])
