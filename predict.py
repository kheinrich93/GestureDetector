import tensorflow as tf
import graph
import numpy as np
from src.tf_utils import prepare_img_for_predict
import os


def predict(sample, dir, hp):
    N_CLASSES = hp.n_classes
    # INPUT_DIM = hp.input_dim
    # SCALE_FACTOR = hp.scale_factor

    # extract labels
    labels = [name for name in os.listdir(
        dir['asl_te']) if os.path.isdir(dir['asl_te'])]
    labels = sorted(labels)

    # # Decodes, scales ,resizes and expands image to fit
    # img_path = dir['asl_te']+'/A/A_test.jpg'
    # sample = prepare_img_for_predict(img_path, SCALE_FACTOR, shape=input_size)

    # Setup network
    model = graph.Net().get_model((64, 64), N_CLASSES)

    # Load weights from pre-trained network
    path = dir['cp_gesture']
    model.load_weights(path).expect_partial()

    # Print prediction with probs
    predictions = model.predict(sample)
    classes = np.argmax(predictions, axis=1)
    prob = np.max(predictions, axis=1)

    print('Predicted letter %s with probability of %f%%' %
          (labels[classes[0]], prob*100))


def predict_from_dir(dir, hp):
    INPUT_DIM = hp.input_dim
    SCALE_FACTOR = hp.scale_factor

    img_path = dir['asl_te']+'/A/A_test.jpg'
    sample = prepare_img_for_predict(img_path, SCALE_FACTOR, shape=INPUT_DIM)

    predict(sample, dir, hp)


def predict_from_image(img, dir, hp):
    INPUT_DIM = hp.input_dim
    SCALE_FACTOR = hp.scale_factor
    img = dir['my_data_te']+'/b_crop.jpg'

    sample = prepare_img_for_predict(img, SCALE_FACTOR, shape=INPUT_DIM)

    predict(sample, dir, hp)
