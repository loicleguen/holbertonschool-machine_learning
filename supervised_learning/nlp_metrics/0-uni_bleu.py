#!/usr/bin/env python3
"""
Module permettant de calculer le score BLEU unigramme d'une phrase.
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calcule le score BLEU unigramme pour une phrase candidate.

    Args:
        references: liste de traductions de référence, où chaque référence
                    est une liste de mots.
        sentence: liste contenant les mots de la phrase candidate proposée.

    Returns:
        Le score BLEU unigramme.
    """
    c_len = len(sentence)
    if c_len == 0:
        return 0.0

    # 1. Calcul de la longueur de référence 'r' la plus proche
    ref_lens = [len(ref) for ref in references]
    # En cas d'égalité de distance, on choisis la longueur la plus petite
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c_len), ref_len))

    # 2. Calcul de la pénalité de concision (BP)
    if c_len > r:
        bp = 1.0
    else:
        bp = np.exp(1 - (r / c_len))

    # 3. Calcul de la précision tronquée des unigrammes
    word_counts = {}
    for word in sentence:
        word_counts[word] = word_counts.get(word, 0) + 1

    clipped_count = 0
    for word, count in word_counts.items():
        max_ref_count = max([ref.count(word) for ref in references])
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / c_len

    return bp * precision
