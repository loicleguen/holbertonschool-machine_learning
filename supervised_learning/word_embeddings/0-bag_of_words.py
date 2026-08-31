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

    Returns:
        tuple: (embeddings, features)
    """
    cleaned_sentences = []
    for sentence in sentences:
        # Passage en minuscules
        text = sentence.lower()
        # Supprime spécifiquement le "'s" (ex: "children's" -> "children")
        text = re.sub(r"'s\b", '', text)
        # Supprime la ponctuation restante (ex: "!", "?", etc.)
        text = re.sub(r'[^\w\s]', '', text)
        # Découpe en mots
        words = text.split()
        cleaned_sentences.append(words)

    if vocab is None:
        all_words = set()
        for words in cleaned_sentences:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        features = vocab

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in features:
                j = features.index(word)
                embeddings[i, j] += 1

    return embeddings, features
