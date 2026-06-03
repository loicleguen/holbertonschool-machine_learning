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
    # On fige toutes les couches du backbone
    base_model.trainable = False

    # Entrree originale en 32x32, redimensionnee pour MobileNetV2
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Resizing(224, 224)(inputs)

    # MobileNetV2 attend des valeurs dans [-1, 1]
    x = K.layers.Rescaling(scale=2.0, offset=-1.0)(x)

    # Extraction des features par le backbone gele
    x = base_model(x, training=False)
    extractor = K.Model(inputs, x)

    # On calcule les features une seule fois pour gagner du temps
    features = extractor.predict(X_train, batch_size=128, verbose=1)

    # Tete de classification entrainnee sur les features extraites
    feature_input = K.Input(shape=(features.shape[1],))

    # Couche dense intermediaire
    x = K.layers.Dense(
        256,
        activation="relu",
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(feature_input)

    # Regularisation pour limiter le surapprentissage
    x = K.layers.Dropout(0.3)(x)

    # Sortie finale pour les 10 classes de CIFAR-10
    outputs = K.layers.Dense(
        10,
        activation="softmax",
        kernel_initializer=K.initializers.he_normal(seed=0)
    )(x)

    model = K.Model(feature_input, outputs)

    # Compilation du modele
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entrainement de la tete de classification
    model.fit(
        features,
        Y_train,
        batch_size=128,
        epochs=15,
        validation_split=0.1,
        verbose=1
    )

    # Reconstruction du modele complet pour la sauvegarde
    full_inputs = K.Input(shape=(32, 32, 3))
    full_x = K.layers.Resizing(224, 224)(full_inputs)
    full_x = K.layers.Rescaling(scale=2.0, offset=-1.0)(full_x)
    full_x = base_model(full_x, training=False)
    full_outputs = model(full_x)
    full_model = K.Model(full_inputs, full_outputs)

    # Compilation finale avant sauvegarde
    full_model.compile(
        optimizer=K.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Sauvegarde du modele final attendu par 0-main.py
    full_model.save("cifar10.h5")
