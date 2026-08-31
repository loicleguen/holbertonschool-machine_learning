#!/usr/bin/env python3
"""
Module to create, build and train a Gensim Word2Vec model.
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim Word2Vec model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for use.
        window (int): Maximum distance between current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        Word2Vec: The trained Gensim Word2Vec model.
    """
    # Dans Gensim: sg=0 correspond à CBOW et sg=1 correspond à Skip-gram
    sg = 0 if cbow else 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
