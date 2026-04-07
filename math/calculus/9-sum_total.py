#!/usr/bin/env python3
"""Module that contains the function summation_i_squared"""

def summation_i_squared(n):
    """Function that calculates the sum of i squared from 1 to n"""
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
