import pytest

from flushot import synthetic


@pytest.fixture(scope="session")
def fake():
    """Small synthetic dataset with the real schema: (X, Y, T, submission format)."""
    return synthetic.make(n_train=400, n_test=120, seed=1)


@pytest.fixture
def data_dir(tmp_path):
    return synthetic.write(tmp_path / "data", n_train=400, n_test=120, seed=1)
