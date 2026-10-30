#!/usr/bin/env python3
"""
Module contenant une fonction pour trier un DataFrame dans l'ordre
chronologique inverse puis le transposer.
"""


def flip_switch(df):
    """
    Trie les données dans l'ordre chronologique inverse (décroissant)
    et transpose le DataFrame résultant.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame trié et transposé.
    """
    return df.sort_index(ascending=False).T
