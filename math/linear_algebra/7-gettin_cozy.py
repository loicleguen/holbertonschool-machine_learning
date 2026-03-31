#!/usr/bin/env python3
"""Contains the function cat_matrices2D."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Module that concatenates two matrices along a specific axis."""
    if axis == 0:
        # Concaténation verticale : même nombre de colonnes
        if len(mat1) == 0:
            return [row[:] for row in mat2]
        if len(mat2) == 0:
            return [row[:] for row in mat1]
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Concaténation horizontale : même nombre de lignes
        if len(mat1) != len(mat2):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
    else:
        return None
