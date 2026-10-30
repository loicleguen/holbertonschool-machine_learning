#!/usr/bin/env python3
"""
Module contenant une fonction pour découper un DataFrame pandas
en sélectionnant des colonnes spécifiques et une ligne sur 60.
"""


def slice(df):
    """
    Extrait les colonnes High, Low, Close et Volume_(BTC) d'un DataFrame
    et sélectionne une ligne sur 60.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame découpé.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
