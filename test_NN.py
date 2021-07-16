import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from src.img_utils import show_sample
from src.dataloader import mnist_data
from src.graph import Net

from sklearn.metrics import accuracy_score


def te_gesture_NN(dirs, hp, dataset, model_weights):
    N_CLASSES = hp.n_classes

    # load dataset
    SCALE_FACTOR = hp.scale_factor

    if dataset == 'mnist':
        path = dirs['mnist_te']+'/sign_mnist_test.csv'

        X_test, y_test = mnist_data(path).testing(hp)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./SCALE_FACTOR)

        samples = datagen.flow(X_test, y_test)

        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    elif dataset == 'asl':
        # preprocess to gray
        pass

    #sample = X_test[1, :, :, :]

    weights_path = os.path.join(dirs['cp'], model_weights, 'gestureNN')

    # Setup network
    model = Net().get_model(N_CLASSES)

    # Load weights from pre-trained network
    model.load_weights(weights_path).expect_partial()

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy'])

    # Print evaluation with prob
    results = model.evaluate(samples)

    print('Testing loss %s with total accuracy of %f%%' %
          (results[0], results[1]*100))

    # Predict the label of the test_images
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    # Accuracy score
    acc = accuracy_score(y_test, pred)
    pass
