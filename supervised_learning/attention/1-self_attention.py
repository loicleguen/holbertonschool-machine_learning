#!/usr/bin/env python3
"""
Module SelfAttention pour le mécanisme d'attention (Bahdanau)
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Calcule l'attention pour la traduction automatique
    selon le mécanisme d'attention de Bahdanau.
    """

    def __init__(self, units):
        """
        Constructeur de la classe

        Args:
            units: entier, nombre d'unités cachées
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Passe les entrées dans le mécanisme d'attention

        Args:
            s_prev: tensor (batch, units) état précédent décodeur
            hidden_states: tensor (batch, input_seq_len, units) sorties
                           de l'encodeur

        Returns:
            context, weights
            - context: tensor (batch, units)
            - weights: tensor (batch, input_seq_len, 1)
        """
        # Ajout d'une dimension temporelle : (batch, 1, units)
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        # Calcul du score d'alignement
        score = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states))
        )

        # Application du Softmax pour obtenir les poids
        weights = tf.nn.softmax(score, axis=1)

        # Calcul du vecteur de contexte
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
