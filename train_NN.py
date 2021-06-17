import os
import tensorflow as tf
import dataloader
import graph
import src.utils as utils
import src.img_utils as img_utils


def gesture_NN(dir, batch_size):
    # params
    n_classes = 29
    img_height = 80
    img_width = 80
    BATCH_SIZE = 64
    VAL_SPLIT = 0.2
    EPOCHS = 1

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, validation_split=VAL_SPLIT)

    train_generator = train_datagen.flow_from_directory(
        'res/training/asl/',
        target_size=(150, 150),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    model = graph.create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    model.fit(train_generator, batch_size=BATCH_SIZE,
              validation_split=VAL_SPLIT, epochs=EPOCHS, verbose=2)

    #test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # train_it = datagen.flow_from_directory(
    # directory='res/training/asl/', class_mode='categorical', batch_size=64)

    # datagen.fit(train_it)
    # fits the model on batches with real-time data augmentation:

    pass


'''
    # load training images and labels 3000 max
    images, labels, label_dict = dataloader.load_images(
        dir['asl_tr'], img_height=img_height, img_width=img_width,  max_examples=1000)

    # train_ds, val_ds = create_dataset(images, labels)

    # train_ds, val_ds = optimize_dataset(train_ds, val_ds, batch_size)

    # images, labels = dataloader.read_images(dir['asl_tr'], max_examples=1)

    val_size = int(len(labels) * 0.2)

    x_val = images[-val_size:]/255.0
    # x_val = tf.expand_dims(x_val, axis=0)
    y_val = labels[-val_size:]

    x_train = images[:-val_size]/255.0
    # x_train = tf.expand_dims(x_train, axis=0)
    y_train = labels[:-val_size]

    # y_train = tf.keras.utils.to_categorical(y_train, num_classes=29)
    # y_val = tf.keras.utils.to_categorical(y_val, num_classes=29)
    # dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # dataset = dataset.map(img_utils._parse_function)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_ds = train_ds.map(img_utils._parse_function)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # val_ds = val_ds.map(img_utils._parse_function)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = utils.configure_for_performance(train_ds, batch_size, AUTOTUNE)
    val_ds = utils.configure_for_performance(val_ds, batch_size, AUTOTUNE)

    # ---------
    cp_callback = create_cp(dir)

    model = graph.create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, batch_size=64,
              epochs=3, verbose=2, callbacks=[cp_callback])
'''


def create_dataset(images, labels, split=0.2):

    val_size = int(len(labels) * split)
    list_ds = tf.data.Dataset.from_tensor_slices((images, labels))

    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    return train_ds, val_ds


def optimize_dataset(train_ds, val_ds, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = utils.configure_for_performance(train_ds, batch_size, AUTOTUNE)
    val_ds = utils.configure_for_performance(val_ds, batch_size, AUTOTUNE)

    return train_ds, val_ds


def create_cp(dir):
    checkpoint_path = dir['cp_gesture']+"/cp-best.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=0)

    return cp_callback
