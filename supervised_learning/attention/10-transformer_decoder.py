#!/usr/bin/env python3
"""
Transformer Decoder module
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Class Decoder to create the decoder for a transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of blocks in the decoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in fully connected layer
            target_vocab (int): Size of target vocabulary
            max_seq_len (int): Maximum sequence length possible
            drop_rate (float): Dropout rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Passes input through the decoder

        Args:
            x (tf.Tensor): Tensor of shape (batch, target_seq_len)
                containing target token IDs
            encoder_output (tf.Tensor): Output tensor of encoder of shape
                (batch, input_seq_len, dm)
            training (bool): Indicates if model is training
            look_ahead_mask (tf.Tensor): Mask for 1st multi head attention
            padding_mask (tf.Tensor): Mask for 2nd multi head attention

        Returns:
            tf.Tensor: Decoder output tensor of shape
                (batch, target_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        # Target embedding scaling by sqrt(dm)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through all N decoder blocks
        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask
            )

        return x
