#!/usr/bin/env python3
"""
Module d'interaction pour répondre aux
questions à partir d'un texte de référence.
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Répond en boucle aux questions posées par l'utilisateur à partir
    d'un document de référence.

    Args:
        reference (str): Le document de référence contenant les réponses.
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input('Q: ')
        if user_input.strip().lower() in exit_commands:
            print('A: Goodbye')
            break

        answer = question_answer(user_input, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
        else:
            print(f'A: {answer}')
