import pytest
from contextlib import contextmanager
from src.helper_check import check_val_split, check_letters
from hp.hyperparams import hyperparams

from test_NN import te_gesture_NN
from src.utils import get_dirs


@contextmanager
def does_not_raise():
    yield

# regression tests


def test_NN_accuracy():
    acc = te_gesture_NN(get_dirs(), hyperparams(),
                        'mnist', 'cp_gesture_mnist')
    assert acc >= 0.8, f'Model accuracy is low (achieving {acc*100}%)'


class TestHyperparamsChecks:
    @pytest.mark.parametrize("split, expectation_split", [
        pytest.param(-1.0, pytest.raises(ValueError)),
        (0.55, does_not_raise()),
        pytest.param(1.0, pytest.raises(ValueError))
    ])
    def test_val_split(self, split, expectation_split):
        with expectation_split:
            assert (check_val_split(split)) is not None

    @pytest.mark.parametrize("value, expectation", [
        ('foobar', does_not_raise()),
        pytest.param('', pytest.raises(TypeError)),
        pytest.param('123456', pytest.raises(TypeError))
    ])
    def test_letters(self, value, expectation):
        with expectation:
            assert (check_letters(value)) is not None
