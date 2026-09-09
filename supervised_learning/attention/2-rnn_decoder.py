#!/usr/bin/env python3
"""
Module RNNDecoder pour le décodeur d'un modèle de traduction automatique
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Décode les séquences d'entrée avec un mécanisme d'attention et un GRU
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Constructeur de la classe

        Args:
            vocab: entier, taille du vocabulaire de sortie
            embedding: entier, dimension du vecteur d'embedding
            units: entier, nombre d'unités cachées dans la cellule RNN
            batch: entier, taille du batch
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Passe les entrées dans la couche du décodeur

        Args:
            x: tensor (batch, 1) index du mot précédent
            s_prev: tensor (batch, units) état caché précédent
            hidden_states: tensor (batch, input_seq_len, units) sorties
                           de l'encodeur

        Returns:
            y, s
            - y: tensor (batch, vocab) vecteur pour le mot prédit
            - s: tensor (batch, units) nouvel état caché du décodeur
        """
        # Calcul des poids et du vecteur de contexte
        context, _ = self.attention(s_prev, hidden_states)

        # Vectorisation du mot x
        x = self.embedding(x)

        # Ajout de la dimension temporelle : (batch, 1, units)
        context_expanded = tf.expand_dims(context, axis=1)

        # Concaténation dans l'ordre : [context_vector, x]
        x = tf.concat([context_expanded, x], axis=-1)

        # Passage dans le GRU
        output, s = self.gru(x)

        # Redimensionnement de la sortie pour le Dense
        output = tf.reshape(output, (-1, output.shape[2]))

        # Projection finale sur le vocabulaire
        y = self.F(output)

        return y, s
