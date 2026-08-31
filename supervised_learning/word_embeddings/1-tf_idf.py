#!/usr/bin/env python3
"""
TF-IDF embedding matrix
"""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix from a list of sentences.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): Predefined list of vocabulary words.
            If None, all unique words from sentences will be used.

    Returns:
        tuple:
            embeddings (numpy.ndarray): Matrix of shape (s, f) containing
                TF-IDF values.
            features (numpy.ndarray): Array of features corresponding to the
                columns of the matrix.
    """
    processed_sentences = []

    for sentence in sentences:
        s = sentence.lower()
        s = re.sub(r"'s\b", "", s)
        s = re.sub(r'[^\w\s]', '', s)
        processed_sentences.append(s.split())

    if vocab is None:
        vocab = sorted(list(set(
            word for s in processed_sentences for word in s
        )))
    else:
        vocab = list(vocab)

    s_count = len(sentences)
    f_count = len(vocab)

    word_to_index = {word: i for i, word in enumerate(vocab)}

    # 1. Calcul de la Fréquence du Mot (TF - Term Frequency)
    tf = np.zeros((s_count, f_count), dtype=float)
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_index:
                tf[i, word_to_index[word]] += 1

    # 2. Calcul de la Fréquence Inverse de
    #   Document (IDF - Inverse Document Frequency)
    # Formule avec smooth_idf=True : log((1 + N) / (1 + df)) + 1
    df = np.count_nonzero(tf, axis=0)
    idf = np.log((1.0 + s_count) / (1.0 + df)) + 1.0

    # 3. Calcul du TF-IDF brut
    tf_idf_matrix = tf * idf

    # 4. Normalisation L2 par ligne (phrase)
    norms = np.linalg.norm(tf_idf_matrix, axis=1, keepdims=True)
    # Évite la division par zéro pour les phrases sans aucun mot du vocabulaire
    norms[norms == 0] = 1.0

    embeddings = tf_idf_matrix / norms

    return embeddings, np.array(vocab)
