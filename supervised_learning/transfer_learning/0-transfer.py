#!/usr/bin/env python3
"""Transfer learning for CIFAR-10 using a Keras application."""
from tensorflow import keras as K


if not hasattr(K.backend, "learning_phase"):
    K.backend.learning_phase = 0


def preprocess_data(X, Y):
    """Pre-process CIFAR-10 data for the model."""
    X_p = X.astype("float32") / 255.0
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


def build_model():
    """Build the transfer-learning model."""
    he = K.initializers.he_normal(seed=0)

    inputs = K.Input(shape=(32, 32, 3))
    x = K.layers.Resizing(224, 224)(inputs)
    x = K.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

    base_model = K.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = K.layers.Dropout(0.3)(x)
    x = K.layers.Dense(256, activation="relu", kernel_initializer=he)(x)
    outputs = K.layers.Dense(10, activation="softmax",
                             kernel_initializer=he)(x)

    model = K.Model(inputs=inputs, outputs=outputs)
    return model, base_model


def unfreeze_last_layers(base_model, n_layers=40):
    """Unfreeze the last layers of the backbone except BatchNormalization."""
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:]:
        if not isinstance(layer, K.layers.BatchNormalization):
            layer.trainable = True


if __name__ == "__main__":
    (X_train, Y_train), _ = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)

    model, base_model = build_model()

    model.compile(optimizer=K.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        K.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                  restore_best_weights=True),
        K.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.2,
                                      patience=2, min_lr=1e-6),
    ]

    model.fit(X_train, Y_train,
              batch_size=64,
              epochs=10,
              validation_split=0.1,
              callbacks=callbacks,
              verbose=1)

    unfreeze_last_layers(base_model, n_layers=40)

    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, Y_train,
              batch_size=64,
              epochs=5,
              validation_split=0.1,
              callbacks=callbacks,
              verbose=1)

    model.save("cifar10.h5")
