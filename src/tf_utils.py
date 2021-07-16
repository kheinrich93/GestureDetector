import os.path
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


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

    train_ds = configure_for_performance(train_ds, batch_size, AUTOTUNE)
    val_ds = configure_for_performance(val_ds, batch_size, AUTOTUNE)

    return train_ds, val_ds


def create_cp(dir):
    cp_dir = os.path.dirname(dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=dir, save_weights_only=True, save_best_only=True, verbose=1)

    return cp_callback, cp_dir


def configure_for_performance(ds, batch_size, AUTOTUNE):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def decode_img(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    return img


def prepare_img_for_predict(img, scale_factor, shape):
    if os.path.exists(img):
        img = decode_img(img)

    img = tf.cast(img, tf.float32) / scale_factor
    img = tf.image.resize(img, shape)
    img = tf.expand_dims(img, axis=0)
    return img


def create_generator_flow_from_dir(dir, datagen, target_size, subset="training", color_mode="rgb", batch_size=32, shuffle=True, class_mode='categorical'):
    gen = datagen.flow_from_directory(
        dir,
        color_mode=color_mode,
        batch_size=batch_size,
        target_size=target_size,
        subset=subset,
        shuffle=shuffle,
        class_mode=class_mode)

    return gen


def summary_to_file(dir, model):
    with open(dir + '/report.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def crop_to_bb(img, offset_height, offset_width, target_height, target_width):
    return tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)


def visualize_results(history):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    ax = axes.flat

    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(ax=ax[0])
    ax[0].set_title("Accuracy", fontsize=15)
    ax[0].set_ylim(0, 1.1)

    pd.DataFrame(history.history)[['loss', 'val_loss']].plot(ax=ax[1])
    ax[1].set_title("Loss", fontsize=15)
    plt.show()

class privateCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\nReached 99%% accuracy")
            self.model.stop_training = True
