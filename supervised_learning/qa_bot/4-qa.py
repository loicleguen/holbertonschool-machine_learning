#!/usr/bin/env python3
"""
Module pour le système de réponse aux questions multi-références.
Combine la recherche sémantique (sélection de document) et le modèle
BERT QA pour répondre aux questions posées sur un corpus de textes.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer

semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Interroge interactivement un corpus de documents pour répondre
    aux questions de l'utilisateur.

    Args:
        corpus_path (str): Chemin vers le dossier contenant les documents
                           de référence (.md).
    """
    # Chargement du tokenizer BERT SQuAD et du modèle QA TensorFlow Hub
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Mots-clés déclenchant la fermeture de l'application
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            question = input("Q: ")
        except (KeyboardInterrupt, EOFError):
            print("\nA: Goodbye")
            break

        # Vérification si l'utilisateur souhaite quitter
        if question.lower().strip() in exit_words:
            print("A: Goodbye")
            break

        # 1. Sélection du document le plus pertinent
        #           via la recherche sémantique
        reference = semantic_search(corpus_path, question)

        # 2. Extraction de la réponse exacte dans le document trouvé
        answer = answer_question(question, reference, tokenizer, model)

        # 3. Affichage de la réponse ou d'un message d'erreur si introuvable
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")


def answer_question(question, reference, tokenizer, model):
    """
    Extrait la réponse exacte à une question depuis un texte de référence
    en utilisant un modèle BERT pré-entraîné.

    Args:
        question (str): La question posée par l'utilisateur.
        reference (str): Le texte de référence dans lequel chercher.
        tokenizer: Le tokenizer BERT pour convertir le texte en tokens.
        model: Le modèle TensorFlow Hub BERT QA.

    Returns:
        str: La réponse extraite sous forme de chaîne de caractères,
             ou None si aucune réponse valide n'est trouvée.
    """
    # Tokenisation de la question et de la référence
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Assemblage du format d'entrée BERT : [CLS] Question [SEP] Reference [SEP]
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
              reference_tokens + ['[SEP]'])
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Masque de segments : 0 pour la question, 1 pour le texte de référence
    token_type_ids = ([0] * (len(question_tokens) + 2) +
                      [1] * (len(reference_tokens) + 1))
    attention_mask = [1] * len(input_word_ids)

    # Ajout d'une dimension de batch (1, N) pour TensorFlow
    input_word_ids = tf.expand_dims(tf.constant(input_word_ids), 0)
    token_type_ids = tf.expand_dims(tf.constant(token_type_ids), 0)
    attention_mask = tf.expand_dims(tf.constant(attention_mask), 0)

    # Prédiction des logits de début et de fin de la réponse
    outputs = model([input_word_ids, attention_mask, token_type_ids])

    # Identification des indices de début et de fin (hors tokens spéciaux)
    short_start = tf.argmax(outputs[0][0][1:-1]) + 1
    short_end = tf.argmax(outputs[1][0][1:-1]) + 1

    # Si l'indice de début dépasse celui de fin, la réponse n'est pas valide
    if short_start > short_end:
        return None

    # Reconstitution de la réponse en chaîne de caractères
    answer_tokens = tokens[short_start:short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer.strip()
