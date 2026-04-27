#!/usr/bin/env python3
"""Trains a model with mini-batch gradient descent,
validation, early stopping, and learning rate decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Trains a model with mini-batch gradient descent, validation,
    early stopping, and learning rate decay"""
    callbacks = []
    # Early stopping
    if early_stopping and validation_data is not None:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(es)
    # Learning rate decay
    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            lr = alpha / (1 + decay_rate * epoch)
            print(
                f"\nEpoch {epoch + 1}: "
                f"LearningRateScheduler setting learning rate to {lr}."
            )
            return lr
        lrs = K.callbacks.LearningRateScheduler(schedule, verbose=0)
        callbacks.append(lrs)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
