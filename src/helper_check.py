import sys
import os
from src.utils import error_msgs
from typing import List
from hp.hyperparams import hyperparams


def check_hp(hp: hyperparams) -> hyperparams:
    check_letters(hp.letters)
    check_val_split(hp.val_split)
    check_scale_factor(hp.scale_factor)
    return hp


def check_letters(letters: List) -> List:
    if False in [letter.isalpha() for letter in letters] or not letters:
        raise TypeError(error_msgs.not_letter + ': ' + letters)
    else:
        return letters


def check_val_split(val_split: float) -> float:
    if val_split >= 1.0 or val_split < 0:
        raise ValueError(
            error_msgs.val_split_value_not_valid + ": " + str(val_split))
    else:
        return float(val_split)


def check_scale_factor(scale_factor: float) -> float:
    if scale_factor > 255.0 or scale_factor <= 0:
        raise ValueError(
            error_msgs.not_a_valid_scale_factor + ": " + str(check_scale_factor))
    else:
        return float(check_scale_factor)


def check_path(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(error_msgs.dir_not_existing + ": " + path)
    else:
        return path


def check_x_to_label_dim(x, y):
    if not x.shape[0] == y.shape[0]:
        raise ValueError(error_msgs.dim_mismatch +
                         ": Input (%d) is not equal to label (%s)" % (x.shape[0], y.shape[0]))
    else:
        return x, y
