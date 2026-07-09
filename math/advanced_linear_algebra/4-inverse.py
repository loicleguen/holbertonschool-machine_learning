#!/usr/bin/env python3
"""Module pour calculer l'inverse d'une matrice carrée."""


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


def inverse(matrix):
    """Calcule l'inverse d'une matrice donnée.

    Args:
        matrix (list of lists): La matrice d'entrée.

    Returns:
        list of lists: La matrice inverse, ou None si elle est singulière.

    Raises:
        TypeError: Si la structure n'est pas une liste de listes.
        ValueError: Si la matrice est vide ou n'est pas carrée.
    """
    # 1. Vérification du type de base
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Si la liste est vide [] -> TypeError selon la gestion des sorties
    if len(matrix) == 0 or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2. Vérification de la validité de la taille (non-vide et carrée)
    n = len(matrix)
    if n == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # 3. Calcul du déterminant principal
    det = determinant(matrix)
    if det == 0:
        return None

    # 4. Cas particulier : Matrice 1x1
    if n == 1:
        return [[1 / matrix[0][0]]]

    # 5. Construction de la matrice des cofacteurs
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            sub_matrix = [
                row[:j] + row[j + 1:]
                for k, row in enumerate(matrix) if k != i
            ]
            minor_val = determinant(sub_matrix)
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor_row.append(minor_val * sign)
        cofactor_matrix.append(cofactor_row)

    # 6. Transposition (Adjugate) et division par le déterminant simultanées
    inverse_matrix = []
    for j in range(n):
        inverse_row = []
        for i in range(n):
            # Division de l'élément transposé par le déterminant global
            inverse_row.append(cofactor_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
