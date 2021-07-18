from typing import Tuple
from numpy import full, ndarray


def create_test_img(dim: Tuple[int, int, int] = (28, 28, 1), value: int = 0) -> ndarray:
    return full(dim, value, dtype=int)
