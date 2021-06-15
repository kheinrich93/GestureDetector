import os
import src.img_utils as img_utils


def load_images(dir):
    data = []
    images = []

    for label in os.listdir(dir):
        for imgs in os.listdir(os.path.join(dir, label)):
            path = os.path.join(dir, label, imgs)
            # image = img_utils.read_image(path)
            # image = img_utils.resize_image_to_size(image, dim=(80, 80))
            # images.append(image)
            data.append([label, path])

    return data
