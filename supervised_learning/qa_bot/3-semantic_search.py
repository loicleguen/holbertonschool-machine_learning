#!/usr/bin/env python3
"""
Performs semantic search on a corpus of documents
"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path (str): path to the corpus of reference documents.
        sentence (str): sentence from which to perform semantic search.

    Returns:
        str: reference text of the document most similar to sentence.
    """
    # Modèle USE Large v5
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(model_url)

    documents = []
    articles = sorted(os.listdir(corpus_path))

    for article in articles:
        if not article.endswith('.md'):
            continue
        file_path = os.path.join(corpus_path, article)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            documents.append(f.read())

    # Génération des embeddings
    embeddings = embed([sentence] + documents)

    # Calcul de la similarité (produit scalaire)
    corr = np.inner(embeddings[0], embeddings[1:])

    # Document avec le score le plus élevé
    closest_idx = np.argmax(corr)

    return documents[closest_idx]
