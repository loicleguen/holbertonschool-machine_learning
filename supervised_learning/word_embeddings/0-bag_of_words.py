#!/usr/bin/env python3
"""
Module pour créer une matrice Bag of Words.
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Crée une matrice d'embedding Bag of Words.

    Args:
        sentences (list): Liste de phrases à analyser.
        vocab (list, optional): Liste des mots du vocabulaire.
            Si None, tous les mots des phrases sont utilisés.

    Returns:
        tuple: (embeddings, features)
            - embeddings: numpy.ndarray de forme (s, f)
            - features: liste des mots clés utilisés
    """
    cleaned_sentences = []
    for sentence in sentences:
        # Remplace les s apostrophe ("children's" -> "children")
        # et extrait uniquement les mots alpha-numériques en minuscules
        words = re.findall(r'\b\w+\b', sentence.lower())
        cleaned_sentences.append(words)

    if vocab is None:
        # Récupère tous les mots uniques et les trie par ordre alphabétique
        all_words = set()
        for words in cleaned_sentences:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = vocab

    # Initialisation de la matrice de forme (nombre de phrases, nombre de mots)
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Remplissage de la matrice avec les fréquences
    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, features
