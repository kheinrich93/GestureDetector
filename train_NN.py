import tensorflow as tf
import graph
import src.tf_utils as tf_utils


def tr_gesture_NN(dir, batch_size):
    # params
    n_classes = 29
    img_dim = (80, 80)
    BATCH_SIZE = batch_size
    VAL_SPLIT = 0.2
    EPOCHS = 1

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=VAL_SPLIT)

    train_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, img_dim, subset="training", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    val_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, img_dim, subset="validation", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    model = graph.create_model()

    model.compile(
        optimizer="adam",
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    cp_callback, cp_dir = tf_utils.create_cp(dir['cp_gesture'])
    model.load_weights(dir['cp_gesture'])

    model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=val_generator, verbose=1, callbacks=[cp_callback])
