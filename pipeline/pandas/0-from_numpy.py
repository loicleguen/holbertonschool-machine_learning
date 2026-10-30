#!/usr/bin/env python3
"""
Module fournissant une fonction pour convertir un tableau NumPy
en DataFrame pandas.
"""
import pandas as pd


def from_numpy(array):
    """
    Crée un pd.DataFrame à partir d'un np.ndarray avec des colonnes
    nommées par ordre alphabétique et en majuscules (A, B, C, ...).

    Args:
        array (np.ndarray): Le tableau NumPy à convertir.

    Returns:
        pd.DataFrame: Le DataFrame pandas créé.
    """
    nb_colonnes = array.shape[1]
    noms_colonnes = [chr(ord('A') + i) for i in range(nb_colonnes)]
    return pd.DataFrame(array, columns=noms_colonnes)
