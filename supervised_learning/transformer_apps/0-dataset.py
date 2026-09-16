#!/usr/bin/env python3
"""
Module 0-dataset
Contient la classe Dataset pour charger et préparer les données de traduction.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Classe Dataset permettant de charger et prétraiter le jeu de données
    Portuguese-to-English pour le modèle Transformer.
    """

    def __init__(self):
        """
        Initialise les jeux de données d'entraînement et de validation,
        puis crée les tokenizers pour le portugais et l'anglais.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Entraîne des sub-word tokenizers à partir des données fournies.

        Args:
            data: un tf.data.Dataset contenant des paquets (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en: les tokenizers entraînés
        """
        pt_base = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        en_base = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_texts = (pt.numpy().decode('utf-8') for pt, en in data)
        en_texts = (en.numpy().decode('utf-8') for pt, en in data)

        vocab_size = 2**13
        tokenizer_pt = pt_base.train_new_from_iterator(
            pt_texts, vocab_size=vocab_size
        )
        tokenizer_en = en_base.train_new_from_iterator(
            en_texts, vocab_size=vocab_size
        )

        return tokenizer_pt, tokenizer_en
