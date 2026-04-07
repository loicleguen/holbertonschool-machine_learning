#!/usr/bin/env python3
"""Module that contains the function poly_derivative"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial:"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [i * poly[i] for i in range(1, len(poly))]
    if derivative == 0:
        return [0]
    return derivative
