#!/usr/bin/env python3
"""
Bag of Words embedding matrix
"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag of Words embedding matrix from a list of sentences.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): A predefined list of vocabulary words
        to use for the analysis. If None, all unique words from sentences will
        be used to build the vocabulary. Defaults to None.

    Returns:
        tuple:
            E (numpy.ndarray): A 2D array of shape (s, f) containing
            word frequencies.
            features (list): The list of features corresponding
            to the columns of the E matrix.
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

    word_to_index = {word: i for i, word in enumerate(vocab)}
    E = np.zeros((len(sentences), len(vocab)), dtype=int)

    for i, s in enumerate(processed_sentences):
        for word in s:
            if word in word_to_index:
                E[i, word_to_index[word]] += 1

    return E, np.array(vocab)
