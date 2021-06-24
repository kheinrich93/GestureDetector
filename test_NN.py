import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import split
from tensorflow.python.ops.numpy_ops.np_array_ops import imag
import graph
import src.tf_utils as tf_utils
from keras.models import load_model
import numpy as np
from src.tf_utils import decode_img
import os


def te_gesture_NN(dir):
    n_classes = 29

    # test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1./255)

    # test_generator = test_gen.flow_from_directory(
    #     dir['asl_te'],
    #     color_mode="rgb",
    #     class_mode='categorical')

    img_path = dir['asl_te']+'/A/A_test.jpg'

    labels = [name for name in os.listdir(
        dir['asl_te']) if os.path.isdir(dir['asl_te'])]

    img = decode_img(img_path)/255.0

    samples = img

    model = graph.Net().get_model(samples.shape, n_classes)

    # model = load_model(dir['cp_gesture'])

    # correct input size, adapt graph for testing
    samples = tf.expand_dims(samples, axis=0)

    predictions = model.predict(samples)
    classes = tf.math.argmax(predictions, axis=1)
    print('Predicted letter %s' % labels[classes[0]])
