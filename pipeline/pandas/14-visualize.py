#!/usr/bin/env python3
"""
Script pour nettoyer, agréger et visualiser les données du DataFrame
à un intervalle quotidien à partir de 2017.
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Suppression de la colonne Weighted_Price
if 'Weighted_Price' in df.columns:
    df = df.drop(columns=['Weighted_Price'])

# 2. Renommer Timestamp en Date
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Conversion du timestamp UNIX en valeur datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Indexer le dataframe sur la colonne Date
df = df.set_index('Date')

# 5. Remplir les valeurs manquantes dans Close avec la valeur précédente
df['Close'] = df['Close'].ffill()

# 6. Remplir High, Low et Open avec la valeur de Close
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])

# 7. Remplir les volumes avec 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 8. Filtrer à partir de 2017 et rééchantillonner
#               à l'intervalle quotidien ('D')
df = df.loc['2017-01-01':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

print(df)

# Trace le graphique
df.plot()
plt.show()
