import os.path

import src.utils as utils
from src.img_utils import read_image as read_image
from src.img_utils import to_grayscale as to_grayscale
import cv2

dir = utils.get_dirs()

img_src = "img"

if img_src == "img":
    img_dir = os.path.join(dir["res"], 'test_ok.jpg')
    cap = read_image(img_dir)
else:
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        gray = to_grayscale(frame)


# semantic hand segmentation

# gesture classification

# software reaction
