#!/usr/bin/env python3
"""Module that contains the function add_matrices"""


def add_matrices(mat1, mat2):
    """Function that adds two matrices element-wise"""
    if type(mat1) is list and type(mat2) is list:
        if len(mat1) != len(mat2):
            return None
        result = [add_matrices(a, b) for a, b in zip(mat1, mat2)]
        if None in result:
            return None
        return result
    return (mat1 + mat2 if type(mat1) in (int, float)
            and type(mat2) in (int, float) else None)
