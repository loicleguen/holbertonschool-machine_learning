#!/usr/bin/env python3
"""Module pour calculer le déterminant d'une matrice carrée."""


def determinant(matrix):
    """Calcule le déterminant d'une matrice de listes.

    Args:
        matrix (list of lists): La matrice carrée d'entrée.

    Returns:
        int ou float: Le déterminant de la matrice.

    Raises:
        TypeError: Si la structure n'est pas une liste de listes.
        ValueError: Si la matrice n'est pas carrée.
    """
    # 1. Vérification des exceptions de type de base
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Cas particulier : [[]] représentent une matrice 0x0
    if matrix == [[]]:
        return 1

    # Vérification que chaque élément est bien une liste
    if len(matrix) == 0 or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 2. Vérification que la matrice est bien carrée
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # 3. Cas de base de la récursion
    # Matrice 1x1
    if n == 1:
        return matrix[0][0]

    # Matrice 2x2 (optimisation de calcul)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # 4. Calcul récursif pour les dimensions supérieures (n >= 3)
    det = 0
    for j in range(n):
        # Création de la sous-matrice en retirant la ligne 0 et la colonne j
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        # Alternance du signe (-1)^j * l'élément * le sous-déterminant
        sign = 1 if j % 2 == 0 else -1
        det += sign * matrix[0][j] * determinant(sub_matrix)

    return det
