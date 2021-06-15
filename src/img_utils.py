import os.path
import cv2


def read_image(img_path):

    if os.path.isfile(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:
        raise SystemExit('Unable to open %s' % img_path)

    return img


def to_grayscale(img_path):
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

    return gray
