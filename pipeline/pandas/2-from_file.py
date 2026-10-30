#!/usr/bin/env python3
"""
Module fournissant une fonction pour charger des données
depuis un fichier dans un DataFrame pandas.
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Charge les données d'un fichier sous forme de pd.DataFrame.

    Args:
        filename (str): Le chemin du fichier à charger.
        delimiter (str): Le séparateur de colonnes dans le fichier.

    Returns:
        pd.DataFrame: Le DataFrame pandas contenant les données chargées.
    """
    return pd.read_csv(filename, sep=delimiter)
