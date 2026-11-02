#!/usr/bin/env python3
"""
Module contenant une fonction pour créer un MultiIndex hiérarchisé par Timestamp
puis par origine de données (bitstamp / coinbase).
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatène deux DataFrames entre les timestamps 1417411980 et 1417417980,
    inverse les niveaux du MultiIndex pour placer 'Timestamp' en premier
    niveau, et trie les données par ordre chronologique.

    Args:
        df1 (pd.DataFrame): Le DataFrame de coinbase.
        df2 (pd.DataFrame): Le DataFrame de bitstamp.

    Returns:
        pd.DataFrame: Le DataFrame concaténé et réorganisé.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df1_filtered = df1_indexed.loc[1417411980:1417417980]
    df2_filtered = df2_indexed.loc[1417411980:1417417980]

    df = pd.concat([df2_filtered, df1_filtered], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel(0, 1)
    df = df.sort_index()

    return df
