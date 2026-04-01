#!/usr/bin/env python3
"""Contains the function cat_matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatène deux matrices le long d'un axe donné"""
    import copy
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    def shape(mat):
        if not isinstance(mat, list):
            return []
        return [len(mat)] + shape(mat[0])

    if axis == 0:
        if len(shape(mat1)) != len(shape(mat2)) or shape(mat1)[1:] != shape(mat2)[1:]:
            return None
        return copy.deepcopy(mat1) + copy.deepcopy(mat2)

    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        merged = cat_matrices(a, b, axis - 1)
        if merged is None:
            return None
        result.append(merged)
    return result
