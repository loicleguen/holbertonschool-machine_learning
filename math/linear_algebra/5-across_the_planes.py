#!/usr/bin/env python3
"""Adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    if len(mat1) != len(mat2):
        return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
        result.append([a + b for a, b in zip(row1, row2)])
    return result
