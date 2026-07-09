#!/usr/bin/env python3
"""Module pour calculer la matrice des mineurs d'une matrice carrée."""


def determinant(matrix):
    """Calcule le déterminant d'une matrice (fonction d'appui)."""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        sign = 1 if j % 2 == 0 else -1
        det += sign * matrix[0][j] * determinant(sub_matrix)
    return det


def minor(matrix):
    """Calcule la matrice des mineurs d'une matrice donnée.

    Args:
        matrix (list of lists): La matrice d'entrée.

    Returns:
        list of lists: La matrice des mineurs.

    Raises:
        TypeError: Si la structure n'est pas une liste de listes.
        ValueError: Si la matrice est vide ou n'est pas carrée.
    """
    # 1. Vérification du type de base
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Si la liste est vide [] -> Ne respecte pas "list of lists" selon l'output
    if len(matrix) == 0 or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2. Vérification de la validité de la taille (non-vide et carrée)
    n = len(matrix)
    if n == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 3. Cas particulier : Matrice 1x1
    # Le mineur d'un élément unique par convention dans ce contexte vaut 1
    if n == 1:
        return [[1]]

    # 4. Construction de la matrice des mineurs
    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            # Créer la sous-matrice en enlevant la ligne i et la colonne j
            sub_matrix = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix) if k != i
            ]
            # Calculer le déterminant de cette sous-matrice
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)

    return minor_matrix
