#!/usr/bin/env python3
"""
Module sdp_attention pour le calcul de la Scaled Dot Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calcule la Scaled Dot Product Attention

    Args:
        Q: tensor de forme (..., seq_len_q, dk) contenant la matrice Query
        K: tensor de forme (..., seq_len_v, dk) contenant la matrice Key
        V: tensor de forme (..., seq_len_v, dv) contenant la matrice Value
        mask: tensor pouvant être diffusé en (..., seq_len_q, seq_len_v)

    Returns:
        output, weights
        - output: tensor de forme (..., seq_len_q, dv)
        - weights: tensor de forme (..., seq_len_q, seq_len_v)
    """
    # Recupération de la dimension dk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Produit matriciel Q x K^T : (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Mise à l'échelle (scaling)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Application du masque optionnel
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax sur le dernier axe pour obtenir les poids
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiplication par Value
    output = tf.matmul(weights, V)

    return output, weights
