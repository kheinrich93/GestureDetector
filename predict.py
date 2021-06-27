import tensorflow as tf
import graph
import numpy as np
from src.tf_utils import decode_img, prepare_img_for_predict
import os


def predict(dir, hp):
    n_classes = hp.n_classes
    input_size = hp.input_dim
    SCALE_FACTOR = hp.scale_factor

    labels = [name for name in os.listdir(
        dir['asl_te']) if os.path.isdir(dir['asl_te'])]
    labels = sorted(labels)

    imgs_path = []
    for dirs in os.listdir(dir['asl_te']):
        path = os.path.join(dir['asl_te'], dirs, dirs+'_test.jpg')
        imgs_path.append(path)

    # Decodes, scales ,resizes and expands image to fit
    img_path = dir['asl_te']+'/A/A_test.jpg'
    sample = prepare_img_for_predict(img_path, SCALE_FACTOR, shape=input_size)

    # Setup network
    model = graph.Net().get_model((64, 64), n_classes)

    # Load weights from pre-trained network
    path = dir['cp_gesture']
    model.load_weights(path).expect_partial()

    # Print prediction with probs
    predictions = model.predict(sample)
    classes = np.argmax(predictions, axis=1)
    prob = np.max(predictions, axis=1)

    print('Predicted letter %s with probability of %f%%' %
          (labels[classes[0]], prob*100))
