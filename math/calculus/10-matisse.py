#!/usr/bin/env python3
"""Module that contains the function poly_derivative"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial:"""
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    return [coef * i for i, coef in enumerate(poly)][1:]
