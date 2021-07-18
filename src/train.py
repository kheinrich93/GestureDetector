import os
import tensorflow as tf

from src.graph import Net
from src.tf_utils import summary_to_file, create_cp
from hp.hyperparams import hyperparams
from keras.callbacks import History


def train(dirs: dict, hp: hyperparams, save_cp: bool, save_weights_as: str, use_pretrained_cp: bool, train_generator: any, val_generator: any, loss: any) -> History:
    N_CLASSES = hp.n_classes
    EPOCHS = hp.epochs

    model = Net().gestureNN(N_CLASSES)

    opt = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['accuracy'])

    # Create and safe summary of model-architecture
    summary_to_file(dirs['summary'], model)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.1, patience=2, verbose=1, mode='auto', min_lr=0.00001)

    # Train network, can use with pretrained weights
    if save_cp:
        checkpoint_cb, _ = create_cp(dirs['cp_gesture'])

        if use_pretrained_cp:
            path = os.path.join(dirs['cp'], save_weights_as, 'gestureNN')
            model.load_weights(path)

        history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator,
                            steps_per_epoch=None, verbose='auto', callbacks=[early_stopping_cb, reduce_lr_cb, checkpoint_cb])

    else:
        history = model.fit(train_generator, epochs=EPOCHS,
                            validation_data=val_generator, steps_per_epoch=None, verbose='auto')

    # Show accuracy, loss plot
    return history
