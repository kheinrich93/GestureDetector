import os.path
from cv2 import imshow, waitKey, imread, IMREAD_COLOR, rectangle, resize, INTER_AREA, cvtColor, COLOR_BGR2GRAY, namedWindow, resizeWindow, WINDOW_NORMAL
import numpy as np


def show_sample(sample):

    if sample.ndim == 4:
        #sample = sample.numpy()
        sample = np.squeeze(sample, axis=0)

    # namedWindow('image', WINDOW_NORMAL)
    # resizeWindow('image', 600, 600)

    imshow("Display window", sample)
    k = waitKey(0)
    return ("displayed image")


def read_image(img_path):

    if os.path.isfile(img_path):
        img = imread(img_path, IMREAD_COLOR)
    else:
        raise SystemExit('Unable to open %s' % img_path)
    return img


def to_grayscale(img_path):
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

    return gray


def resize_image_to_percent(img, scale_percent=60):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return resize(img, dim, interpolation=INTER_AREA)


def resize_image_to_size(img, dim=(200, 200)):

    return resize(img, dim)


def draw_bb(img, start_point, end_point, color=(255, 0, 0), thickness=2):

    return rectangle(img, start_point, end_point, color, thickness)
