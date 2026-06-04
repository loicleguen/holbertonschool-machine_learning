#!/usr/bin/env python3
"""Transfer learning for CIFAR-10 using a Keras application."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Pre-process CIFAR-10 data for the model.

    Args:
        X: images CIFAR-10 de forme (m, 32, 32, 3)
        Y: labels CIFAR-10 de forme (m,)

    Returns:
        X_p: images normalisees
        Y_p: labels en one-hot
    """
    # On ramene les pixels dans l'intervalle [0, 1]
    X_p = X.astype("float32") / 255.0

    # On transforme les labels entiers en vecteurs one-hot
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


if __name__ == "__main__":
    K.utils.set_random_seed(0)

    # Chargement du jeu de donnees CIFAR-10
    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)

    # Backbone pre-entraine sur ImageNet
    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False

    # Modele complet: entree 32x32 -> redimensionnement -> backbone -> tete
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Resizing(224, 224)(inputs)
    x = K.layers.Rescaling(scale=2.0, offset=-1.0)(x)
    x = base_model(x, training=False)

    x = K.layers.Dense(
        256,
        activation="relu",
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    model = K.Model(inputs, outputs)

    # Phase 1: entrainement de la tete seulement
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train,
        Y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: fine-tuning leger des dernieres couches du backbone
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    for layer in base_model.layers[-20:]:
        if isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=K.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        Y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarde du modele final
    model.save("cifar10.h5")
