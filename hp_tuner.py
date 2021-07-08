import tensorflow as tf
from src.graph import Net
import src.tf_utils as tf_utils

import kerastuner as kt


def find_opt_hp(dir, batch_size):
    # params
    n_classes = 29
    img_dim = (80, 80)
    BATCH_SIZE = batch_size
    VAL_SPLIT = 0.2
    EPOCHS = 2

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=VAL_SPLIT)

    train_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, img_dim, subset="training", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    val_generator = tf_utils.create_generator_flow_from_dir(
        dir['asl_tr'], datagen, img_dim, subset="validation", color_mode="rgb", batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

    tuner = kt.Hyperband(Net.create_exp_model,
                         objective='val_accuracy',
                         max_epochs=EPOCHS,
                         directory='hp_results',
                         project_name='hp_tuner')

    tuner.search(train_generator, epochs=EPOCHS, validation_data=val_generator)
    best_model = tuner.get_best_models()[0]
