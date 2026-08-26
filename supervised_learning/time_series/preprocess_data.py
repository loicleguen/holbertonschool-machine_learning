#!/usr/bin/env python3
"""
Preprocesses Coinbase and Bitstamp CSV data for BTC time series forecasting.
"""
import numpy as np
import pandas as pd


def clean_and_resample(file_path):
    """
    Cleans raw CSV data and resamples 60-second intervals into 1-hour data.
    """
    df = pd.read_csv(file_path)

    # Conversion du Timestamp Unix en datetime et définition comme index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.set_index('Timestamp', inplace=True)

    # Remplissage des valeurs manquantes de prix
    df['Close'] = df['Close'].ffill()
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])

    # Les volumes manquants sont mis à zéro
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    df['Weighted_Price'] = df['Weighted_Price'].fillna(df['Close'])

    # Rééchantillonnage toutes les heures (1H)
    df_resampled = df.resample('1h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }).dropna()

    return df_resampled


def preprocess():
    """
    Main preprocessing function that cleans, scales,
    and formats sequences for model training.
    """
    # Chargement du fichier Bitstamp extrait
    bitstamp_path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    df = clean_and_resample(bitstamp_path)

    # Sélection des fonctionnalités utiles (Close price + volumes)
    data = df[['Close', 'Volume_(BTC)', 'Weighted_Price']].values

    # Normalisation min-max
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    scaled_data = (data - min_val) / (max_val - min_val)

    # Séquences : 24h d'historique (X) pour prédire Close à t+1 (Y)
    X, Y = [], []
    input_steps = 24

    for i in range(len(scaled_data) - input_steps):
        X.append(scaled_data[i:i + input_steps])
        Y.append(scaled_data[i + input_steps, 0])

    X = np.array(X)
    Y = np.array(Y)

    # Séparation Train (80%) / Validation (20%)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # Sauvegarde compacte
    np.savez_compressed(
        'preprocessed_btc.npz',
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        min_val=min_val,
        max_val=max_val
    )
    print("Prétraitement terminé.")


if __name__ == '__main__':
    preprocess()
