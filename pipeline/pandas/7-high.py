#!/usr/bin/env python3
"""
Module contenant une fonction pour trier un DataFrame pandas
selon les valeurs de la colonne High par ordre décroissant.
"""


def high(df):
    """
    Trie le DataFrame d'entrée par la colonne 'High' en ordre décroissant.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame trié par la colonne 'High'.
    """
    return df.sort_values(by='High', ascending=False)
