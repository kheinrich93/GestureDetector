import os


def load_images(dir):
    images = []
    labels = []

    for file in os.listdir(dir):
        labels.append(file)

    return [images, labels]
