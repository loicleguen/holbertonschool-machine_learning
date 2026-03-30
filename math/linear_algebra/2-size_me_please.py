#!/usr/bin/env python3
"""Defines the function matrix_shape(matrix)"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Args:
        matrix (list): The matrix to calculate the shape of

    Returns:
        list: The shape of the matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        return []
    if type(matrix[0]) is not list:
        return [len(matrix)]
    return [len(matrix)] + matrix_shape(matrix[0])
