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


def resize_image_to_percent(img, scale_percent=60):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def resize_image_to_size(img, dim=(200, 200)):

    return cv2.resize(img, dim)


def draw_bb(img, start_point, end_point, color=(255, 0, 0), thickness=2):

    return cv2.rectangle(img, start_point, end_point, color, thickness)
