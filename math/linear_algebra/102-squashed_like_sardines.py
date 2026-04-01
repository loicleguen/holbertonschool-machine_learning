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
        if len(mat) == 0:
            return [0]
        return [len(mat)] + shape(mat[0])
    if shape(mat1) != shape(mat2):
        if axis == 0 and shape(mat1)[1:] == shape(mat2)[1:]:
            pass
        else:
            return None
    if axis == 0:
        return (
            [copy.deepcopy(row) for row in mat1] +
            [copy.deepcopy(row) for row in mat2]
        )
    if len(mat1) != len(mat2):
        return None
    result = []
    for a, b in zip(mat1, mat2):
        merged = cat_matrices(a, b, axis=axis-1)
        if merged is None:
            return None
        result.append(copy.deepcopy(merged))
    return result
