#!/usr/bin/env python3
"""
Module contenant une fonction pour sélectionner des colonnes spécifiques
d'un DataFrame pandas et les convertir en tableau NumPy.
"""


def array(df):
    """
    Sélectionne les 10 dernières lignes des colonnes 'High' et 'Close'
    et les convertit en un np.ndarray.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée contenant les colonnes
                           'High' et 'Close'.

    Returns:
        numpy.ndarray: Le tableau NumPy contenant les 10 dernières lignes
                       des colonnes sélectionnées.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
