#!/usr/bin/env python3
"""
Transformer Decoder Block module
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class DecoderBlock to create a decoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        Args:
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in fully connected layer
            drop_rate (float): Dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu'
        )
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Passes input through the decoder block

        Args:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len, dm)
            encoder_output (tf.Tensor): Output tensor of encoder of shape
                (batch, input_seq_len, dm)
            training (bool): Indicates if model is training
            look_ahead_mask (tf.Tensor): Mask for 1st multi head attention
            padding_mask (tf.Tensor): Mask for 2nd multi head attention

        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm)
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        dense_out = self.dense_hidden(out2)
        dense_out = self.dense_output(dense_out)
        dense_out = self.dropout3(dense_out, training=training)
        out3 = self.layernorm3(out2 + dense_out)

        return out3
