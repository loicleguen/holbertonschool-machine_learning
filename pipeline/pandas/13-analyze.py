#!/usr/bin/env python3
"""
Module contenant une fonction pour calculer des statistiques descriptives
sur les colonnes d'un DataFrame pandas à l'exception de la colonne Timestamp.
"""


def analyze(df):
    """
    Calcule les statistiques descriptives pour toutes les colonnes
    d'un DataFrame à l'exception de la colonne 'Timestamp'.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant les statistiques
                       descriptives (count, mean, std, min, 25%, etc.).
    """
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    return df.describe()
