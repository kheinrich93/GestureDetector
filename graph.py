import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow import keras


def create_model():

    model = keras.Sequential()

    model.add(Conv2D(filters=64, kernel_size=5, padding='same',
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

    model.compile(
        optimizer="adam",
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
