#!/usr/bin/env python3
"""
Module de réponse aux questions (QA) basé sur BERT.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Trouve un extrait de texte dans un document de référence pour répondre
    à une question donnée.

    Args:
        question (str): La question posée.
        reference (str): Le document de référence.

    Returns:
        str: La réponse extraite du texte, ou None si aucune n'est trouvée.
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Encodage de la question et de la référence
    inputs = tokenizer(question, reference, return_tensors='tf')
    input_word_ids = inputs['input_ids']
    input_mask = inputs['attention_mask']
    input_type_ids = inputs['token_type_ids']

    # Prédiction des logits de début et de fin de réponse
    outputs = model([input_word_ids, input_mask, input_type_ids])
    start_logits, end_logits = outputs[0], outputs[1]

    # Extraction des indices des tokens avec les plus fortes probabilités
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]

    # Aucun résultat valide si l'indice de fin précède l'indice de début
    if start_index > end_index:
        return None

    # Conversion des tokens extraits en texte rémanent
    tokens = tokenizer.convert_ids_to_tokens(
        input_word_ids[0][start_index:end_index + 1]
    )
    answer = tokenizer.convert_tokens_to_string(tokens)

    if not answer.strip():
        return None

    return answer
