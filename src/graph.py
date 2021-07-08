import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras import Sequential, optimizers
from tensorflow.keras.models import Model


class Net:

    @staticmethod
    def conv2D_block(x, filters, kernel_size=5, kernel_regularizer=None, name=''):
        x = Conv2D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, activation=None,
                   padding='same', name=name)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def get_model(self, n_classes):
        filters = 32

        kernel_regularizer = None

        inputs = Input(shape=(None, None, 1), name='input')

        x1 = self.conv2D_block(inputs, filters*2, name='conv1')
        x2 = self.conv2D_block(x1, filters*2, name='conv2')

        pool1 = MaxPooling2D(pool_size=(4, 4))(x2)
        drop1 = Dropout(0.5)(pool1)

        x3 = self.conv2D_block(drop1, filters*4, name='conv3')
        x4 = self.conv2D_block(x3, filters*6, name='conv4')

        pool2 = MaxPooling2D(pool_size=(4, 4))(x4)
        drop2 = Dropout(0.5)(pool2)

        x5 = self.conv2D_block(drop2, filters*8, name='conv5')

        drop3 = Dropout(0.5)(x5)

        #flatten = Flatten()(drop3)
        flatten = tf.keras.layers.GlobalAveragePooling2D()(drop3)

        predictions = Dense(n_classes, activation='softmax')(flatten)

        return Model(inputs=inputs, outputs=predictions)


# TODO: convert to func model
def create_exp_model(hp):

    hp_filters = hp.Int('filter', min_value=8, max_value=128, step=8)

    model = Sequential()

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

    model.compile(optimizer=optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model
