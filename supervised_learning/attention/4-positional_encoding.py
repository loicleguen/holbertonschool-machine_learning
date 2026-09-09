#!/usr/bin/env python3
"""
Module positional_encoding pour le calcul du codage positionnel
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calcule le codage positionnel pour un modèle Transformer

    Args:
        max_seq_len: entier, longueur maximale de la séquence
        dm: entier, profondeur du modèle (dimension des embeddings)

    Returns:
        numpy.ndarray de forme (max_seq_len, dm) contenant les
        vecteurs de codage positionnel
    """
    PE = np.zeros((max_seq_len, dm))

    # Tableau des positions : forme (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Terme d'échelle pour la fréquence : forme (1, dm)
    div_term = np.exp(
        np.arange(0, dm, 2) * -(np.log(10000.0) / dm)
    )

    # Application du sinus aux indices pairs (2i)
    PE[:, 0::2] = np.sin(position * div_term)

    # Application du cosinus aux indices impairs (2i + 1)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
