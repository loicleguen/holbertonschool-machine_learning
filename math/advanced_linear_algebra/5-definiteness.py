#!/usr/bin/env python3
"""Module pour déterminer la nature d'une matrice (definiteness)."""
import numpy as np


def definiteness(matrix):
    """Calcule la propriété de définition (definiteness) d'une matrice.

    Args:
        matrix (np.ndarray): Matrice carrée symétrique.

    Returns:
        str: "Positive definite", "Positive semi-definite",
             "Negative semi-definite", "Negative definite", ou "Indefinite".
             Renvoie None si la matrice n'est pas valide.

    Raises:
        TypeError: Si l'entrée n'est pas un numpy.ndarray.
    """
    # 1. Vérification du type imposé
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # 2. Vérification de la validité de la matrice (2D et carrée)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Cas d'une matrice vide de type np.array([]) ou np.array([[]])
    if matrix.size == 0:
        return None

    # 3. Vérification de la symétrie de la matrice
    # Une matrice non symétrique n'est pas valide pour ce test classique
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calcul des valeurs propres (eigvalsh est optimisé pour les matrices
        # symétriques réelles et garantit des valeurs purement réelles)
        eigenvalues = np.linalg.eigvalsh(matrix)
    except np.linalg.LinAlgError:
        return None

    # 4. Analyse des signes des valeurs propres
    # Tolérance pour éviter les erreurs d'arrondi de float autour de zéro
    tol = 1e-10

    pos = np.any(eigenvalues > tol)
    neg = np.any(eigenvalues < -tol)
    zero = np.any(np.abs(eigenvalues) <= tol)

    if pos and not neg and not zero:
        return "Positive definite"
    if pos and not neg and zero:
        return "Positive semi-definite"
    if neg and not pos and not zero:
        return "Negative definite"
    if neg and not pos and zero:
        return "Negative semi-definite"
    if pos and neg:
        return "Indefinite"

    return None
