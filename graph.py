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

    return model
