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
            - embeddings: numpy.ndarray de forme (s, f)
            - features: liste des features utilisées
    """
    # Nettoyage et découpage des mots pour chaque phrase
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.lower().split()
        cleaned_words = []
        for word in words:
            # Conserve uniquement les caractères alphanumériques
            cleaned = re.sub(r'\W+', '', word)
            if cleaned:
                cleaned_words.append(cleaned)
        cleaned_sentences.append(cleaned_words)

    if vocab is None:
        # Récupère les mots uniques et les trie par ordre alphabétique
        all_words = set()
        for words in cleaned_sentences:
            all_words.update(words)
        features = sorted(list(all_words))
    else:
        # Si vocab est fourni, il devient directement la liste des features
        features = list(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # Dictionnaire de correspondance mot -> indice dans features
    vocab_dict = {word: idx for idx, word in enumerate(features)}

    # Remplissage de la matrice
    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in vocab_dict:
                j = vocab_dict[word]
                embeddings[i, j] += 1

    return embeddings, features
