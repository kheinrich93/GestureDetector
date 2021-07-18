from typing import Tuple

import pandas as pd
import numpy as np

from hp.hyperparams import hyperparams
from src.helper_check import *


class mnist_data:
    def __init__(self, path: str):
        self.csv_df = pd.read_csv(check_path(path))

    def training(self, hp: hyperparams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        LETTERS = check_letters(hp.letters)
        VAL_SPLIT = check_val_split(hp.val_split)
        SHUFFLE = hp.training_shuffle

        # creates sparse categorical int labels
        letters = [letter.lower() for letter in LETTERS]
        letter_2_number = [ord(char) - 97 for char in letters]

        # remove unused letters
        mask = self.csv_df['label'].isin(letter_2_number)
        csv_df = self.csv_df[mask]

        # shuffle
        if SHUFFLE:
            csv_df = csv_df.sample(frac=1.0).reset_index(drop=True)

        # Split into training, test and validation sets
        train_df, val_df = val_split_df(csv_df, VAL_SPLIT)

        # Create labels for training and validation set
        y_train = create_label(train_df)
        y_val = create_label(val_df)

        # Reshape the training and test set to use them with a generator
        X_train = reshape_mnist_to_img(train_df)
        X_val = reshape_mnist_to_img(val_df)

        # Test shape of train and val
        assert X_train.shape[0] == y_train.shape[0], "Row numbers of X and y data must be identical"
        assert X_val.shape[0] == y_val.shape[0], "Row numbers of X and y data must be identical"

        return X_train, y_train, X_val, y_val

    def testing(self, hp: hyperparams) -> Tuple[np.ndarray, np.ndarray]:
        LETTERS = check_letters(hp.letters)

        # Creates sparse categorical int labels
        letters = [letter.lower() for letter in LETTERS]
        letter_2_number = [ord(char) - 97 for char in letters]

        # Remove unused letters
        mask = self.csv_df['label'].isin(letter_2_number)
        test_df = self.csv_df[mask]

        y_test = create_label(test_df)
        X_test = reshape_mnist_to_img(test_df)

        return X_test, y_test

    def predict(self) -> np.ndarray:
        X_predict = reshape_mnist_to_img(self.csv_df)
        return X_predict


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    return pd.CategoricalIndex(df['label']).codes


def reshape_mnist_to_img(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop('label', axis=1).values.reshape(df.shape[0], 28, 28, 1)


def val_split_df(df: pd.DataFrame, val_split: float) -> Tuple[np.ndarray, np.ndarray]:
    val_index = int(df.shape[0]*val_split)

    train_df = df.iloc[val_index:]
    val_df = df.iloc[:val_index]

    return train_df, val_df
