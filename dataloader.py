import os
import src.img_utils as img_utils
import tensorflow as tf


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

    return images, labels, label_dict
