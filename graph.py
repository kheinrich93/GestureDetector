import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow import keras


def conv2D_block(model, filters, kernel_regularizer):
    model.add(Conv2D(filters=filters, kernel_size=5,
                     padding='same', activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(BatchNormalization())

    return model


def create_model(img_dim, n_classes):

    kernel_regularizer = tf.keras.regularizers.L1()

    filters = 32

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(input_shape=(img_dim[0], img_dim[1], 3)))

    model = conv2D_block(model, filters, kernel_regularizer)
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))

    model = conv2D_block(model, filters*2, kernel_regularizer)
    model.add(Dropout(0.2))

    model = conv2D_block(model, filters*2, kernel_regularizer)
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))

    model = conv2D_block(model, filters*4, kernel_regularizer)
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


def create_exp_model(hp):

    hp_filters = hp.Int('filter', min_value=8, max_value=128, step=8)

    model = keras.Sequential()

    model.add(Conv2D(filters=hp_filters, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=5,
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=128, kernel_size=5,
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=5,
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=256, kernel_size=5,
                     padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(29, activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model
