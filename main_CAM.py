from cv2 import imshow, waitKey, imwrite

from src.img_utils import read_image, draw_bb
from src.utils import get_dirs
from src.tf_utils import crop_to_bb, decode_img
from predict import predict_from_image
from hp.hyperparams import hyperparams

import tensorflow as tf


# set dirs and hyperparams
dirs = get_dirs()
hp = hyperparams()

# img src cam or img
img_src = 'img'

if img_src == 'img':
    path = dirs['my_data_te']+'/v_test.jpg'
    img = read_image(path)


# crop window in bb
box_len = 450
start_point = (10, 200)

# convert to tf image
img_cropped = crop_to_bb(
    img, start_point[1], start_point[0], box_len, box_len)
img_cropped = tf.image.resize(img_cropped, (200, 200))

imwrite(dirs['res']+'/b_crop.jpg', img_cropped.numpy())

# predict_from_image
predict_from_image(img_cropped, dirs, hp)

# plot crop
img_bb = draw_bb(img, start_point,
                 (start_point[0]+box_len, start_point[1]+box_len))
# imshow("Display window", img_bb)
# k = waitKey(0)
