#!/usr/bin/env python3
"""
Module contenant une fonction pour supprimer les lignes contenant
des valeurs NaN dans la colonne Close d'un DataFrame pandas.
"""


def prune(df):
    """
    Supprime toutes les lignes où la colonne 'Close' a une valeur NaN.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame nettoyé.
    """
    return df.dropna(subset=['Close'])
