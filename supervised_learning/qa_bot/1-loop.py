#!/usr/bin/env python3
"""
Script qui crée une boucle d'interaction utilisateur simples.
"""


def main():
    """
    Exécute la boucle interactive et s'arrête si l'utilisateur saisit
    'exit', 'quit', 'goodbye', ou 'bye' (insensible à la casse).
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input('Q: ')
        if user_input.strip().lower() in exit_commands:
            print('A: Goodbye')
            break
        print('A:')


if __name__ == '__main__':
    main()
