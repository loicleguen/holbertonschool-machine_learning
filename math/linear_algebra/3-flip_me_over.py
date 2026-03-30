#!/usr/bin/env python3
"""Defines the function matrix_transpose(matrix)"""


def matrix_transpose(matrix):
    """Retourne la transposée d'une matrice 2D"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
