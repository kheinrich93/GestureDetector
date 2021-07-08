import os
import tensorflow as tf
import pandas as pd
import numpy as np

import src.img_utils as img_utils


def load_images(dir, img_height, img_width, max_examples):
    labels = []
    images = []
    label_dict = {"label_str": [], "label_nr": []}

    for idx, label in enumerate(os.listdir(dir)):
        print("Loading data from", label)
        for idimg, imgs in enumerate(os.listdir(os.path.join(dir, label))):
            if idimg < max_examples:
                path = os.path.join(dir, label, imgs)
                image = img_utils.decode_img(path, img_height, img_width)
                labels.append(tf.one_hot(idx, 29))
                images.append(image)
            else:
                break
        print("... done")
        label_dict["label_str"].append(label)
        label_dict["label_nr"].append(idx)

    #images = np.array(image)
    images = tf.cast(images, tf.float32)
    labels = np.array(labels).astype("int32")

    return images, labels, label_dict


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

    #img_path = tf.convert_to_tensor(img_path, dtype=tf.string)
    #labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    images = np.array(img_path)
    labels = np.array(labels).astype("int32")

    return img_path, labels


def load_mnist_csv(dirs, hp):
    LETTERS = hp.letters
    VAL_SPLIT = hp.val_split

    train_df = pd.read_csv(dirs['mnist_tr']+'/sign_mnist_train.csv')

    # creates categorical int labels
    letters = ['A', 'B', 'C', 'H', 'K', 'L', 'O']
    letters = [letter.lower() for letter in LETTERS]
    letter_2_number = [ord(char) - 97 for char in letters]

    # remove unused letters
    mask = train_df['label'].isin(letter_2_number)
    train_df = train_df[mask]

    # shuffle
    train_df = train_df.sample(frac=1.0).reset_index(drop=True)

    # Split into training, test and validation sets
    val_index = int(train_df.shape[0]*VAL_SPLIT)

    train_df_original = train_df.copy()

    train_df = train_df_original.iloc[val_index:]
    val_df = train_df_original.iloc[:val_index]

    # Create labels for training and validation set
    y_train = train_df['label']
    y_train = pd.CategoricalIndex(y_train).codes
    y_val = val_df['label']
    y_val = pd.CategoricalIndex(y_val).codes

    # Reshape the training and test set to use them with a generator
    X_train = train_df.drop('label', axis=1).values.reshape(
        train_df.shape[0], 28, 28, 1)
    X_val = val_df.drop('label', axis=1).values.reshape(
        val_df.shape[0], 28, 28, 1)

    print(X_train.shape, X_val.shape)

    return X_train, y_train, X_val, y_val
