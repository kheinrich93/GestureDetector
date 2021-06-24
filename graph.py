import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras import Sequential
from tensorflow.keras.models import Model
# make class of net https://www.tensorflow.org/guide/checkpoint#loading_mechanics


class Net:
    @staticmethod
    def conv2D_block(x, filters, kernel_regularizer, dropout=0, kernel_size=5, name=''):
        x = Conv2D(filters, kernel_size=kernel_size, kernel_regularizer=kernel_regularizer, activation=None,
                   padding='same', name=name)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        return x

    def get_model(self, img_dim, n_classes):
        filters = 32

        kernel_regularizer = tf.keras.regularizers.L1()

        inputs = Input(shape=(img_dim[0], img_dim[1], 3), name='input')

        conv1 = self.conv2D_block(
            inputs, filters, kernel_regularizer, dropout=0, kernel_size=5, name='CONV1')
        pool1 = MaxPooling2D(pool_size=(4, 4))(conv1)

        conv2 = self.conv2D_block(
            pool1, filters*2, kernel_regularizer, dropout=0.5, kernel_size=5, name='CONV2')
        pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)

        conv3 = self.conv2D_block(
            pool2, filters*4, kernel_regularizer, dropout=0.5, kernel_size=5, name='CONV3')

        flatten = Flatten()(conv3)
        predictions = Dense(n_classes, activation='softmax')(flatten)

        return Model(inputs=inputs, outputs=predictions)


def seq_conv2D_block(model, filters, kernel_regularizer):
    model.add(Conv2D(filters=filters, kernel_size=5,
                     padding='same', activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(BatchNormalization())

    return model


def seq_create_model(img_dim, n_classes):

    kernel_regularizer = tf.keras.regularizers.L1()

    filters = 32

    model = Sequential()

    model.add(tf.keras.layers.InputLayer(
        input_shape=(img_dim[0], img_dim[1], 3)))

    model = seq_conv2D_block(model, filters, kernel_regularizer)
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model = seq_conv2D_block(model, filters*2, kernel_regularizer)
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model = seq_conv2D_block(model, filters*2, kernel_regularizer)
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))

    model = seq_conv2D_block(model, filters*4, kernel_regularizer)
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

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model
