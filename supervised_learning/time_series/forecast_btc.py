#!/usr/bin/env python3
"""
Builds, trains, and validates a Keras model for BTC forecasting using tf.data.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_model(input_shape):
    """
    Constructs an RNN architecture using LSTM layers.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Prédit la valeur du cours de clôture
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


def main():
    """
    Main function to load data, feed tf.data.Dataset, train, and evaluate.
    """
    # 1. Chargement des données pré-traitées
    data = np.load('preprocessed_btc.npz')
    X_train, Y_train = data['X_train'], data['Y_train']
    X_val, Y_val = data['X_val'], data['Y_val']

    # 2. Création de la pipeline tf.data.Dataset
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 3. Construction et entraînement du modèle
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    print("Début de l'entraînement...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks
    )

    # 4. Sauvegarde du modèle
    model.save('btc_forecast_model.h5')
    print("Entraînement terminé et modèle sauvegardé.")


if __name__ == '__main__':
    main()
