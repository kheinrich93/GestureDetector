from src.graph import Net
import numpy as np


def predict(hp, samples, weights_path):
    N_CLASSES = hp.n_classes
    LABELS = hp.letters

    # Setup network
    model = Net().get_model(N_CLASSES)

    # Load weights from pre-trained network
    model.load_weights(weights_path).expect_partial()

    # show_sample(sample)

    # Print prediction with probs
    predictions = model.predict(samples)

    classes = np.argmax(predictions, axis=1)
    prob = np.max(predictions, axis=1)

    # print('Predicted letter %s with probability of %f%%' %
    #       (LABELS[classes[0]], prob*100))
