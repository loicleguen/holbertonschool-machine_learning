#!/usr/bin/env python3
"""
Module 1-dataset
Contient la classe Dataset avec la méthode d'encodage des phrases.
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Classe Dataset permettant de charger, prétraiter
        et encoder le jeu de données
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

    def encode(self, pt, en):
        """
        Encode une paire de phrases de traduction en tokens.

        Args:
            pt: tf.Tensor contenant la phrase en portugais
            en: tf.Tensor contenant la phrase en anglais

        Returns:
            pt_tokens, en_tokens: listes de tokens incluant les tokens
            de début (vocab_size) et de fin (vocab_size + 1)
        """
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens
