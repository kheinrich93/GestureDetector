import tensorflow as tf
from tensorflow import keras
import graph
import src.tf_utils as tf_utils
import os


def tr_gesture_NN(dir, hp, use_pretrained_cp=False, save_cp=True):
    # hyperparams
    N_CLASSES = hp.n_classes
    IMG_DIM = hp.input_dim
    BATCH_SIZE = hp.batch_size
    VAL_SPLIT = hp.val_split
    EPOCHS = hp.epochs
    SCALE_FACTOR = hp.scale_factor

    # Load images to flow
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./SCALE_FACTOR, validation_split=VAL_SPLIT)

    train_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, target_size=IMG_DIM, subset="training", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    val_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, target_size=IMG_DIM, subset="validation", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    # Create and compile model
    model = graph.Net().get_model(IMG_DIM, N_CLASSES)

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    # Output summary as txt in /summary
    tf_utils.summary_to_file(dir['summary'], model)

    # Train network, can start with cp-weights
    if save_cp:
        cp_callback, cp_dir = tf_utils.create_cp(dir['cp_gesture'])

        if use_pretrained_cp:
            path = dir['cp_gesture']
            model.load_weights(path)

        model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=val_generator, steps_per_epoch=None, verbose='auto', callbacks=[cp_callback])

    else:
        model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=val_generator, steps_per_epoch=100, verbose='auto')
