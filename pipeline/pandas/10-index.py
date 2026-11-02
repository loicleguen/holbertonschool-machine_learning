#!/usr/bin/env python3
"""
Module contenant une fonction pour définir la colonne Timestamp
comme index d'un DataFrame pandas.
"""


def index(df):
    """
    Définit la colonne 'Timestamp' comme index du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame modifié avec l'index mis à jour.
    """
    return df.set_index('Timestamp')
