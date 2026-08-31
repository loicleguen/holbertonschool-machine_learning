#!/usr/bin/env python3
"""
Module to convert a Gensim Word2Vec model to a Keras Embedding layer.
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model: A trained Gensim Word2Vec model.

    Returns:
        keras.layers.Embedding: The trainable Keras Embedding layer.
    """
    # Extraction de la matrice de poids de la couche wv de Gensim
    weights = model.wv.vectors

    # Création de la couche Embedding initialisée avec la matrice de poids
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )

    return embedding_layer
