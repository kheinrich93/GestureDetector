import tensorflow as tf
import os.path


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
    checkpoint_path = dir + "/cp-best.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)

    return cp_callback


def configure_for_performance(ds, batch_size, AUTOTUNE):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def decode_img(img_path, img_height=64, img_width=64):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    # tf.io.decode_and_crop_jpeg

    return tf.image.resize(img, [img_height, img_width])


def tf_crop(img, offset):
    img = tf.image.crop_to_bounding_box(img, offset[0], offset[1], 200, 200)
    return img


def create_generator_flow_from_dir(dir, datagen, img_dim, subset="training", color_mode="rgb", batch_size=32, shuffle=True, class_mode='categorical'):
    gen = datagen.flow_from_directory(
        dir,
        color_mode=color_mode,
        batch_size=batch_size,
        target_size=(img_dim[0], img_dim[1]),
        subset=subset,
        shuffle=shuffle,
        class_mode=class_mode)

    return gen
