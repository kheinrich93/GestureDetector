import os
import tensorflow as tf
import pandas as pd
import numpy as np


class mnist_data():
    def __init__(self, path):
        self.csv_df = pd.read_csv(path)

    def training(self, hp):
        LETTERS = hp.letters
        VAL_SPLIT = hp.val_split
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

        return X_train, y_train, X_val, y_val

    def testing(self, hp):
        LETTERS = hp.letters

        # creates sparse categorical int labels
        letters = [letter.lower() for letter in LETTERS]
        letter_2_number = [ord(char) - 97 for char in letters]

        # remove unused letters
        mask = self.csv_df['label'].isin(letter_2_number)
        test_df = self.csv_df[mask]

        y_test = create_label(test_df)
        X_test = reshape_mnist_to_img(test_df)

        return X_test, y_test

    def predict(self):
        X_predict = reshape_mnist_to_img(self.csv_df)
        return X_predict


def create_label(df):
    labels = df['label']
    return pd.CategoricalIndex(labels).codes


def reshape_mnist_to_img(df):
    return df.drop('label', axis=1).values.reshape(df.shape[0], 28, 28, 1)


def val_split_df(df, val_split):
    val_index = int(df.shape[0]*val_split)

    train_df = df.iloc[val_index:]
    val_df = df.iloc[:val_index]

    return train_df, val_df


def read_images(dir, max_examples):
    img_path = []
    labels = []

    for idx, label in enumerate(os.listdir(dir)):
        print("Loading data from", label)
        for idimg, imgs in enumerate(os.listdir(os.path.join(dir, label))):
            if idimg < max_examples:
                path = os.path.join(dir, label, imgs)
                img_path.append(path)
                labels.append(idx)
            else:
                break

    images = np.array(img_path)
    labels = np.array(labels).astype("int32")

    return img_path, labels
