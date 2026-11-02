#!/usr/bin/env python3
"""
Module contenant une fonction pour concaténer deux DataFrames pandas
indexés sur leur Timestamp avec des clés hiérarchiques.
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexe deux DataFrames sur leur colonne Timestamp, extrait les lignes
    de df2 jusqu'au timestamp 1417411920 inclus, puis les concatène au-dessus
    de df1 avec des clés hiérarchiques ('bitstamp' et 'coinbase').

    Args:
        df1 (pd.DataFrame): Le DataFrame de coinbase.
        df2 (pd.DataFrame): Le DataFrame de bitstamp.

    Returns:
        pd.DataFrame: Le DataFrame final concaténé.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed.loc[:1417411920]

    return pd.concat(
        [df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
