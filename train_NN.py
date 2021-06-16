import os
import tensorflow as tf
import dataloader
import graph
import src.utils as utils


def gesture_NN(dir, batch_size):
    # load training images and labels 3000 max
    images, labels, label_dict = dataloader.load_images(
        dir['asl_tr'], img_height=70, img_width=70,  max_examples=1)

    train_ds, val_ds = create_dataset(images, labels)

    train_ds, val_ds = optimize_dataset(train_ds, val_ds, batch_size)

    cp_callback = create_cp(dir)

    model = graph.create_model()

    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, batch_size=batch_size,
              epochs=3, verbose=2, callbacks=[cp_callback])


def create_dataset(images, labels, split=0.2):

    val_size = int(len(labels) * split)
    list_ds = tf.data.Dataset.from_tensor_slices((images, labels))

    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    return train_ds, val_ds


def optimize_dataset(train_ds, val_ds, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_ds = utils.configure_for_performance(train_ds, batch_size, AUTOTUNE)
    val_ds = utils.configure_for_performance(val_ds, batch_size, AUTOTUNE)

    return train_ds, val_ds


def create_cp(dir):
    checkpoint_path = dir['cp_gesture']+"/cp-best.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=0)

    return cp_callback
