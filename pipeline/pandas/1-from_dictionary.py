#!/usr/bin/env python3
"""
Script qui crée un DataFrame pandas à partir d'un dictionnaire.
"""
import pandas as pd

columns = {
    'First': [0.0, 0.5, 1.0, 1.5], 'Second': ['one', 'two', 'three', 'four']
    }

df = pd.DataFrame(columns, index=['A', 'B', 'C', 'D'])
