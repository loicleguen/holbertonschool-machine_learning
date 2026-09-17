#!/usr/bin/env python3
"""
Module 4-create_masks
Contient la fonction create_masks pour générer les masques
d'attention (padding et look-ahead) pour le modèle Transformer.
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Crée les masques d'attention pour l'entraînement et la validation.

    Args:
        inputs: tf.Tensor de forme (batch_size, seq_len_in)
                contenant la phrase d'entrée
        target: tf.Tensor de forme (batch_size, seq_len_out)
                contenant la phrase cible

    Returns:
        encoder_mask: masque de padding pour l'encodeur
                      de forme (batch_size, 1, 1, seq_len_in)
        combined_mask: masque combiné (padding + look-ahead) pour le décodeur
                      de forme (batch_size, 1, seq_len_out, seq_len_out)
        decoder_mask: masque de padding pour le deuxième bloc d'attention
                      du décodeur de forme (batch_size, 1, 1, seq_len_in)
    """
    # Masque de padding pour l'encodeur (inputs)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Masque de padding pour le décodeur (inputs) dans le deuxième bloc
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Masque de padding pour la cible (target)
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Masque Look-ahead pour cacher les tokens futurs
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )

    # Masque combiné : maximum entre le look-ahead et le padding du target
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
