#!/usr/bin/env python3
"""
Module RNN Encoder pour la traduction automatique
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encode les séquences d'entrée pour la traduction automatique
    à l'aide d'une couche GRU.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Constructeur de la classe

        Args:
            vocab: entier, taille du vocabulaire d'entrée
            embedding: entier, dimension du vecteur d'embedding
            units: entier, nombre d'unités cachées dans la cellule RNN
            batch: entier, taille du batch
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initialise les états cachés pour la cellule RNN avec des zéros

        Returns:
            Tensor de forme (batch, units) contenant les états cachés
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Passe les entrées à travers la couche d'encodage

        Args:
            x: tensor de forme (batch, input_seq_len) contenant les mots
            initial: tensor de forme (batch, units) avec l'état initial

        Returns:
            outputs, hidden
            - outputs: tensor de forme (batch, input_seq_len, units)
            - hidden: tensor de forme (batch, units) (dernier état caché)
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
