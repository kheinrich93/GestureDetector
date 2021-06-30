import tensorflow as tf
import graph
import src.tf_utils as tf_utils
import numpy as np
import os
import pandas as pd
from src.utils import get_imgdata_from_csv
import string


def tr_gesture_NN(dir, hp, use_pretrained_cp=False, save_cp=True):
    # hyperparams
    N_CLASSES = hp.n_classes
    IMG_DIM = hp.input_dim
    BATCH_SIZE = hp.batch_size
    VAL_SPLIT = hp.val_split
    EPOCHS = hp.epochs
    SCALE_FACTOR = hp.scale_factor
    COLOR_MODE = hp.color_mode

    # mnist_images, mnist_labels = get_imgdata_from_csv(
    #     dir['mnist_tr']+'/sign_mnist_train.csv')

    # letters = ['a', 'b', 'c', 'h', 'k', 'l', 'o']
    # letter_2_number = [ord(char) - 97 for char in letters]

    # images_filtered = []
    # labels_filtered = np.array([], dtype=int)
    # # get all images with label in list
    # for idx, label in enumerate(mnist_labels):
    #     if label in letter_2_number:
    #         images_filtered.append(mnist_images[idx])
    #         labels_filtered = np.append(
    #             labels_filtered, letter_2_number.index(label))
    #         # labels_filtered.append(letter_2_number.index(label))

    # # integer_mapping = {x: i for i, x in enumerate(letters)}
    # # vec = [integer_mapping[word] for word in letters]

    # images_filtered = np.expand_dims(images_filtered, axis=3)

    # print(images_filtered.shape)
    # print(labels_filtered.shape)

    # mnist_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rescale=1 / 255)

    # val_mnist_generator = mnist_datagen.flow(
    #     x=images_filtered,
    #     y=labels_filtered,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True
    # )

    # Load images to flow

    tr_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./SCALE_FACTOR, validation_split=VAL_SPLIT, brightness_range=[0.3, 0.9], zoom_range=0.1, shear_range=0.1, width_shift_range=0.1, fill_mode='nearest', rotation_range=10)

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./SCALE_FACTOR, validation_split=VAL_SPLIT)

    # zoom_range=0.2,shear_range=0.2,width_shift_range=0.2,shear_range=0.2,rotation_range=20
    train_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], tr_datagen, target_size=IMG_DIM, subset="training", color_mode=COLOR_MODE, batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    val_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], val_datagen, target_size=IMG_DIM, subset="validation", color_mode=COLOR_MODE, batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    # set correct dims for grayscale
    if COLOR_MODE == 'grayscale':
        IMG_DIM = [IMG_DIM[0], IMG_DIM[1], 1]
    else:
        IMG_DIM = [IMG_DIM[0], IMG_DIM[1], 3]

    # Create and compile model
    model = graph.Net().get_model(IMG_DIM, N_CLASSES)

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    # Output summary as txt in /summary
    tf_utils.summary_to_file(dir['summary'], model)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', min_lr=0.00001)

    # Train network, can start with cp-weights
    if save_cp:
        checkpoint_cb, _ = tf_utils.create_cp(dir['cp_gesture'])

        if use_pretrained_cp:
            path = os.path.join(dir['cp'], 'cp_gesture_7n', 'gestureNN')
            model.load_weights(path)

        history = model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=val_generator, steps_per_epoch=None, verbose='auto', callbacks=[early_stopping_cb, reduce_lr_cb, checkpoint_cb])

    else:
        history = model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS,
                            validation_data=val_generator, steps_per_epoch=100, verbose='auto')
