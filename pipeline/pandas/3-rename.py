#!/usr/bin/env python3
"""
Module contenant une fonction pour renommer, convertir et filtrer
les colonnes d'un DataFrame pandas.
"""
import pandas as pd


def rename(df):
    """
    Renomme la colonne 'Timestamp' en 'Datetime', convertit ses valeurs
    en objets datetime et ne conserve que les colonnes 'Datetime' et 'Close'.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée contenant 'Timestamp'.

    Returns:
        pd.DataFrame: Le DataFrame modifié.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
