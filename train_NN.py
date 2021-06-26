import tensorflow as tf
from tensorflow import keras
import graph
import src.tf_utils as tf_utils
import os


def tr_gesture_NN(dir, batch_size, use_pretrained_cp=False, save_cp=True):
    # params
    n_classes = 29
    img_dim = (64, 64)
    BATCH_SIZE = batch_size
    VAL_SPLIT = 0.2
    EPOCHS = 4

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=VAL_SPLIT)

    train_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, target_size=img_dim, subset="training", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    val_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, target_size=img_dim, subset="validation", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    # create and compiles model
    #model = graph.create_model(img_dim, n_classes)

    model = graph.Net().get_model_exp(img_dim, n_classes)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    tf_utils.summary_to_file(dir['summary'], model)

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
