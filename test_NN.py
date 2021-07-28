import os

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

from hp.hyperparams import hyperparams
from src.dataloader import load_dataset
from src.graph import Net


def te_gesture_NN(dirs: dict, hp: hyperparams, dataset: str, model_weights: str) -> float:
    # load dataset
    loss, X_test, y_test = load_dataset(dataset, dirs, hp)

    # sample = X_test[1, :, :, :]

    weights_path = os.path.join(
        dirs['cp'], model_weights, 'gestureNN')

    # Setup network
    model = Net().gestureNN(hp.n_classes)

    # Load weights from pre-trained network
    model.load_weights(weights_path).expect_partial()

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy'])

    # Predict the label of the test_images
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis=1)

    # Accuracy score
    acc = accuracy_score(y_test, pred)
    print('Accuracy: ', acc*100, '%')

    return acc
