#!/usr/bin/env python3
"""
Module contenant une fonction pour imputer les valeurs manquantes
et supprimer une colonne dans un DataFrame pandas.
"""


def fill(df):
    """
    Nettoie et remplit les valeurs manquantes d'un DataFrame :
    - Supprime la colonne 'Weighted_Price'.
    - Remplit les NaN de 'Close' avec la valeur précédente.
    - Remplit les NaN de 'High', 'Low', 'Open' avec la valeur de 'Close'.
    - Remplit les NaN de 'Volume_(BTC)' et 'Volume_(Currency)' avec 0.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.

    Returns:
        pd.DataFrame: Le DataFrame modifié.
    """
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].ffill()

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
