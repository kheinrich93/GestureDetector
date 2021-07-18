import sys
import os
from src.utils import error_msgs
from typing import List


def check_path(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(error_msgs.dir_not_existing + ": " + path)
    else:
        return path


def check_letters(letters: List) -> List:
    if False in [letter.isalpha() for letter in letters]:
        raise TypeError(error_msgs.not_letter + ': ' + letters)
    else:
        return letters


def check_val_split(val_split: float) -> float:
    # Check type and boundaries of val_split

    if val_split >= 1.0 or val_split < 0:
        raise ValueError(
            error_msgs.val_split_value_not_valid + ": " + str(val_split))
    else:
        return float(val_split)


def check_x_to_label_dim(x, y):
    if not x.shape[0] == y.shape[0]:
        raise ValueError(error_msgs.dim_mismatch +
                         ": Input (%d) is not equal to label (%s)" % (x.shape[0], y.shape[0]))
    else:
        return x, y
