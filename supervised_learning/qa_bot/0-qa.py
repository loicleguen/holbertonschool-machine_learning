#!/usr/bin/env python3
"""
Write a function that finds a snippet of text
within a reference document to answer a question.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Find a snippet of text within a reference document to answer a
    question.

    Args:
        question (str): The question to answer.
        reference (str): The reference document containing the answer.

    Returns:
        str: The extracted answer, or None if no answer is found.
    """
    # Loading the tokenizer for BERT
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Loading the BERT model from TensorFlow Hub
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    # Tokenization of the question and reference text
    question_tokens = tokenizer.tokenize(question)
    reference_tokens = tokenizer.tokenize(reference)

    # Creating input tensors for the model
    tokens = (['[CLS]'] + question_tokens + ['[SEP]'] + reference_tokens
              + ['[SEP]'])
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Creating segment IDs: 0 for question tokens and 1 for reference
    # tokens
    segment_ids = ([0] * (len(question_tokens) + 2)
                   + [1] * (len(reference_tokens) + 1))

    # Converting lists to tensors and adding batch dimension
    input_ids_tensor = [input_ids]
    input_mask_tensor = [input_mask]
    segment_ids_tensor = [segment_ids]

    # Running the model to get the start and end logits
    outputs = model(
        [input_ids_tensor, input_mask_tensor, segment_ids_tensor]
    )

    # Extracting the start and end logits from the model outputs
    start_logits = outputs[0]
    end_logits = outputs[1]

    # Estimating the start and end indices of the answer in the reference
    # text (excluding [CLS] token at index 0)
    start_index = tf.argmax(start_logits[0][1:], axis=-1) + 1
    end_index = tf.argmax(end_logits[0][1:], axis=-1) + 1

    # If the start index is greater than the end index, no valid answer was
    # found
    if start_index > end_index:
        return None

    # Extracting the answer tokens from the reference text using the
    # estimated indices
    answer_tokens = tokens[start_index:end_index + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if not answer.strip():
        return None

    return answer
