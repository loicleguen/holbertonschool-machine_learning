#!/usr/bin/env python3
"""
Module 5-train
Entraîne un modèle Transformer pour la traduction Portugais -> Anglais.
"""
import gc
import tensorflow as tf

# Permet d'allouer la VRAM dynamiquement au lieu de tout bloquer au démarrage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Planificateur dynamique du taux d'apprentissage pour Adam."""

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Calcule la perte SparseCategoricalCrossentropy sans le padding."""
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss_ = loss_object(real, pred)
    mask = tf.cast(
        tf.math.logical_not(tf.math.equal(real, 0)),
        dtype=loss_.dtype
    )
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Calcule la précision en ne tenant pas compte des tokens masqués."""
    accuracies = tf.equal(
        tf.cast(real, tf.int64),
        tf.cast(tf.argmax(pred, axis=-1), tf.int64)
    )
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Crée et entraîne un modèle Transformer.
    """
    data = Dataset(batch_size, max_len)

    input_vocab_size = data.tokenizer_pt.vocab_size + 2
    target_vocab_size = data.tokenizer_en.vocab_size + 2

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    transformer = Transformer(
        N, dm, h, hidden, input_vocab_size, target_vocab_size, max_len
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # Étape d'entraînement compilée
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int64),
        tf.TensorSpec(shape=[None, None], dtype=tf.int64)
    ])
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp,
                tar_inp,
                True,
                enc_mask,
                combined_mask,
                dec_mask
            )
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(
            loss, transformer.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables)
        )

        acc = accuracy_function(tar_real, predictions)
        train_loss(loss)
        train_accuracy(acc)

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inp, tar) in enumerate(data.data_train):
            # S'assurer que les entrées sont bien du type attendu (int64)
            inp = tf.cast(inp, tf.int64)
            tar = tf.cast(tar, tf.int64)

            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    f"Epoch {epoch + 1}, batch {batch}: "
                    f"loss {train_loss.result():.4f} "
                    f"accuracy {train_accuracy.result():.4f}"
                )

        print(
            f"Epoch {epoch + 1}: "
            f"loss {train_loss.result():.4f} "
            f"accuracy {train_accuracy.result():.4f}"
        )

        gc.collect()

    return transformer
