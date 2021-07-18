import sys
import os
from src.utils import error_msgs
from typing import List


def check_path(path: str) -> str:
    if not os.path.isfile(path):
        sys.exit(error_msgs.dir_not_existing + ": " + path)
    else:
        return path


def check_letters(letters: List) -> List:
    if False in [letter.isalpha() for letter in letters]:
        sys.exit(error_msgs.not_letter + ': ' + letters)
    else:
        return letters


def check_val_split(val_split: any) -> float:
    # Check type and boundaries of val_split
    if not isinstance(val_split, float) and (val_split > 1.0 or val_split < 0):
        sys.exit(error_msgs.val_split_value_not_valid + ": " + val_split)
    else:
        return float(val_split)
