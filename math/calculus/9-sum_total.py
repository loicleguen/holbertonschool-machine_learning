#!/usr/bin/env python3
"""Module that contains the function summation_i_squared"""

def summation_i_squared(n):
    """Function that calculates the sum of i squared from 1 to n"""
    if not isinstance(n, int) or n < 1:
        return None
    return sum(i ** 2 for i in range(1, n + 1))
