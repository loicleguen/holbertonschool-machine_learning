#!/usr/bin/env python3
"""
Multi Head Attention module using TensorFlow Keras
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform Multi Head Attention inheriting from tf.keras.layers.Layer
    """

    def __init__(self, dm, h):
        """
        Class constructor

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (h, depth) and transpose
        to shape (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Performs Multi Head Attention

        Args:
            Q (tf.Tensor): Query input of shape (batch, seq_len_q, dk)
            K (tf.Tensor): Key input of shape (batch, seq_len_v, dk)
            V (tf.Tensor): Value input of shape (batch, seq_len_v, dv)
            mask (tf.Tensor): Mask tensor, defaults to None

        Returns:
            output (tf.Tensor): Scaled dot product attention output
            weights (tf.Tensor): Attention weights
        """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)  # (batch, seq_len_q, dm)
        k = self.Wk(K)  # (batch, seq_len_v, dm)
        v = self.Wv(V)  # (batch, seq_len_v, dm)

        q = self.split_heads(q, batch_size)  # (batch, h, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch, h, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  # (batch, h, seq_len_v, depth)

        # Scaled Dot Product Attention
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)

        # Transpose back: (batch, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate heads back into shape: (batch, seq_len_q, dm)
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm)
        )

        # Final linear projection
        output = self.linear(concat_attention)

        return output, attention_weights
