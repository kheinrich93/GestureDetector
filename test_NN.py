import tensorflow as tf
import graph
import src.tf_utils as tf_utils


def te_gesture_NN(dir, batch_size):
    BATCH_SIZE = batch_size

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    test_generator = test_gen.flow_from_directory(
        dir['asl_te'],
        color_mode="rgb",
        class_mode='categorical')

    model = graph.create_model()

    model.compile(
        optimizer="adam",
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    # Loads the weights
    model.load_weights(dir['cp_gesture'])

    # Re-evaluate the model
    loss, acc = model.evaluate(
        test_generator, batch_size=BATCH_SIZE, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
