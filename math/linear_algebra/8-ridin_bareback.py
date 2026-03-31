#!/usr/bin/env python3
"""Contains the function mat_mul"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1) == 0 or len(mat2) == 0 or len(mat1[0]) != len(mat2):
        return None
    result = []
    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            s = 0
            for k in range(len(mat2)):
                s += row[k] * mat2[k][j]
            new_row.append(s)
        result.append(new_row)
    return result
