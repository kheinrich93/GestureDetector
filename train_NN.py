import tensorflow as tf

from src.tf_utils import visualize_results, create_generator_flow_from_dir
from src.dataloader import mnist_data
from src.train import train

from src.helper_check import check_path
from src.utils import error_msgs

from hp.hyperparams import hyperparams


def tr_gesture_NN(dirs: dict, hp: hyperparams, use_pretrained_cp: bool = False, save_weights_as: str = 'gestureWeights', save_cp: bool = False, dataset: str = 'mnist') -> str:
    # hyperparams
    IMG_DIM = hp.input_dim
    BATCH_SIZE = hp.batch_size
    VAL_SPLIT = hp.val_split
    SCALE_FACTOR = hp.scale_factor
    COLOR_MODE = hp.color_mode

    # Create generators of dataset for training
    if dataset == 'mnist':

        path = check_path(dirs['mnist_tr']+'/sign_mnist_train.csv')

        X_train, y_train, X_val, y_val = mnist_data(path).training(hp)

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./SCALE_FACTOR, zoom_range=0.1, shear_range=0.1, fill_mode='nearest', rotation_range=10)

        train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
        val_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

        # label not one hot encoded so sparse
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    elif dataset == 'asl':

        tr_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./SCALE_FACTOR, validation_split=VAL_SPLIT, zoom_range=0.1, shear_range=0.1, width_shift_range=0.1, fill_mode='nearest', rotation_range=10)

        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./SCALE_FACTOR, validation_split=VAL_SPLIT)

        train_generator = create_generator_flow_from_dir(
            dirs['asl_tr'], tr_datagen, target_size=IMG_DIM, subset="training", color_mode=COLOR_MODE, batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

        val_generator = create_generator_flow_from_dir(
            dirs['asl_tr'], val_datagen, target_size=IMG_DIM, subset="validation", color_mode=COLOR_MODE, batch_size=BATCH_SIZE, shuffle=True, class_mode='categorical')

        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        raise SystemExit(error_msgs.dataset_not_dir)

    # Train network
    # test types, filename is string, hp val split okay, x_train = y_train und val
    history = train(dirs, hp, save_cp, save_weights_as, use_pretrained_cp,
                    train_generator, val_generator, loss=loss)

    # Show accuracy, loss plot
    visualize_results(history)

    return 'Training successful'
