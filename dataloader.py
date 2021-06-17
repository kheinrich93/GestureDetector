import os
import numpy as np
import tensorflow as tf
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
