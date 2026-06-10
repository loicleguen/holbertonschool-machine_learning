#!/usr/bin/env python3
"""Transfer Learning on CIFAR-10 using MobileNetV2."""

from tensorflow import keras as K

# Active la désérialisation des Lambda layers si la version de Keras le permet.
# Si l'attribut n'existe pas, on ignore l'erreur.
try:
    K.config.enable_unsafe_deserialization()
except AttributeError:
    pass


def preprocess_data(X, Y):
    """Pre-processes the data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 data
        Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels

    Returns:
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    # Normalise les pixels selon ce que MobileNetV2 attend
    X_p = K.applications.mobilenet_v2.preprocess_input(X)

    # Transforme les labels en one-hot
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Chargement des données CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Modèle pré-entraîné sur ImageNet
    base_model = K.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3),
        pooling="avg"
    )
    base_model.trainable = False

    # Extraction des features une seule fois
    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Resizing(96, 96)(inputs)
    x = base_model(x, training=False)
    extractor = K.Model(inputs, x)

    X_train_features = extractor.predict(X_train, batch_size=64, verbose=1)
    X_test_features = extractor.predict(X_test, batch_size=64, verbose=1)

    # Classifieur sur les features extraites
    feature_input = K.Input(shape=X_train_features.shape[1:])
    x = K.layers.Dense(256, activation="relu")(feature_input)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(10, activation="softmax")(x)
    model = K.Model(feature_input, x)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_features,
        Y_train,
        epochs=20,
        validation_data=(X_test_features, Y_test),
        callbacks=[
            K.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # Modèle complet sauvegardé
    full_inputs = K.Input(shape=(32, 32, 3))
    full_x = K.layers.Resizing(96, 96)(full_inputs)
    full_x = base_model(full_x, training=False)
    full_outputs = model(full_x)
    full_model = K.Model(full_inputs, full_outputs)

    full_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    full_model.save("cifar10.h5")
