#!/usr/bin/env python3
"""
Transformer Network module
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Class Transformer to create a full transformer network
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor

        Args:
            N (int): Number of blocks in the encoder and decoder
            dm (int): Dimensionality of the model
            h (int): Number of heads
            hidden (int): Number of hidden units in fully connected layers
            input_vocab (int): Size of input vocabulary
            target_vocab (int): Size of target vocabulary
            max_seq_input (int): Maximum sequence length for input
            max_seq_target (int): Maximum sequence length for target
            drop_rate (float): Dropout rate
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Passes inputs and targets through the transformer network

        Args:
            inputs (tf.Tensor): Tensor of shape (batch, input_seq_len)
            target (tf.Tensor): Tensor of shape (batch, target_seq_len)
            training (bool): Indicates if model is training
            encoder_mask (tf.Tensor): Padding mask applied to the encoder
            look_ahead_mask (tf.Tensor): Look ahead mask applied to decoder
            decoder_mask (tf.Tensor): Padding mask applied to the decoder

        Returns:
            tf.Tensor: Transformer output tensor of shape
                (batch, target_seq_len, target_vocab)
        """
        enc_output = self.encoder(inputs, training, encoder_mask)

        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask
        )

        final_output = self.linear(dec_output)

        return final_output
