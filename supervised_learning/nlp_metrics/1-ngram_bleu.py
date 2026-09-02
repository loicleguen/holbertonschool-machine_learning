#!/usr/bin/env python3
"""
Module permettant de calculer le score BLEU n-gramme d'une phrase.
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calcule le score BLEU n-gramme pour une phrase candidate.

    Args:
        references: liste de traductions de référence (chaque référence est
                    une liste de mots).
        sentence: liste des mots de la phrase candidate proposée.
        n: taille du n-gramme à évaluer.

    Returns:
        Le score BLEU n-gramme.
    """
    c_len = len(sentence)
    if c_len == 0:
        return 0.0

    # 1. Calcul de la longueur de référence 'r' la
    #       plus proche (basé sur unigrammes)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c_len), ref_len))

    # 2. Calcul de la pénalité de concision (BP)
    if c_len > r:
        bp = 1.0
    else:
        bp = np.exp(1 - (r / c_len))

    # Si la phrase est plus courte que n, la précision vaut 0
    if c_len < n:
        return 0.0

    # 3. Extraction des n-grammes de la phrase candidate sous forme de tuples
    sentence_ngrams = [
        tuple(sentence[i:i + n]) for i in range(c_len - n + 1)
    ]

    # Comptage des fréquences des n-grammes dans la phrase candidate
    ngram_counts = {}
    for ngram in sentence_ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    # 4. Comptage tronqué (clipped count) par rapport aux références
    clipped_count = 0
    for ngram, count in ngram_counts.items():
        max_ref_count = 0
        for ref in references:
            if len(ref) >= n:
                ref_ngrams = [
                    tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
                ]
                ref_count = ref_ngrams.count(ngram)
                if ref_count > max_ref_count:
                    max_ref_count = ref_count
        clipped_count += min(count, max_ref_count)

    # 5. Calcul de la précision $P_n$ (nombre
    #       total de n-grammes = c_len - n + 1)
    total_ngrams = c_len - n + 1
    precision = clipped_count / total_ngrams

    return bp * precision
