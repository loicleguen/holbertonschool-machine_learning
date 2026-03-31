#!/usr/bin/env python3
"""Module that contains the function add_arrays"""


def add_arrays(arr1, arr2):
    """Function that adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
