#!/usr/bin/env python3
"""
Module to create, build and train a Gensim FastText model.
"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim FastText model.

    Args:
        sentences (list): List of sentences to be trained on.
        vector_size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for use.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between current and predicted word.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        gensim.models.FastText: The trained Gensim FastText model.
    """
    sg = 0 if cbow else 1

    model = gensim.models.FastText(
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

    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )

    return model
