import pytest
from contextlib import contextmanager
from src.helper_check import check_val_split
# test return obj type von scores and range


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize("value, expectation", [
    pytest.param(-1.0, pytest.raises(ValueError)),
    (0.55, does_not_raise()),
    pytest.param(1.0, pytest.raises(ValueError))
])
def test_val_split(value, expectation):
    with expectation:
        assert (check_val_split(value)) is not None
