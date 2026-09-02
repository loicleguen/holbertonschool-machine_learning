#!/usr/bin/env python3
"""
Module permettant de calculer le score BLEU n-gramme cumulatif d'une phrase.
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calcule le score BLEU n-gramme cumulatif pour une phrase candidate.

    Args:
        references: liste de traductions de référence (chaque référence est
                    une liste de mots).
        sentence: liste des mots de la phrase candidate proposée.
        n: taille maximale du n-gramme à évaluer.

    Returns:
        Le score BLEU n-gramme cumulatif.
    """
    c_len = len(sentence)
    if c_len == 0:
        return 0.0

    # 1. Calcul de la longueur de référence 'r' la plus proche
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c_len), ref_len))

    # 2. Calcul de la pénalité de concision (BP)
    if c_len > r:
        bp = 1.0
    else:
        bp = np.exp(1 - (r / c_len))

    # Poids égaux pour chaque ordre de k-gramme (1 à n)
    weights = [1 / n] * n
    precisions = []

    # 3. Calcul de la précision $P_k$ pour chaque $k$ de 1 à $n$
    for k in range(1, n + 1):
        if c_len < k:
            precisions.append(0.0)
            continue

        # Extraction des k-grammes de la phrase candidate
        sentence_ngrams = [
            tuple(sentence[i:i + k]) for i in range(c_len - k + 1)
        ]

        ngram_counts = {}
        for ngram in sentence_ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        # Comptage tronqué (clipped count)
        clipped_count = 0
        for ngram, count in ngram_counts.items():
            max_ref_count = 0
            for ref in references:
                if len(ref) >= k:
                    ref_ngrams = [
                        tuple(ref[i:i + k]) for i in range(len(ref) - k + 1)
                    ]
                    ref_count = ref_ngrams.count(ngram)
                    if ref_count > max_ref_count:
                        max_ref_count = ref_count
            clipped_count += min(count, max_ref_count)

        total_ngrams = c_len - k + 1
        precisions.append(clipped_count / total_ngrams)

    # Si une des précisions est nulle, le produit géométrique vaut 0
    if 0 in precisions:
        return 0.0

    # 4. Calcul de la moyenne géométrique pondérée des précisions
    log_precisions = [w * np.log(p) for w, p in zip(weights, precisions)]
    geo_mean = np.exp(sum(log_precisions))

    return bp * geo_mean
