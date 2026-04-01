#!/usr/bin/env python3
"""Contains the function np_slice"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    slices = [slice(*axes.get(i, (None, None, None)))
              for i in range(matrix.ndim)]
    return matrix[tuple(slices)]
