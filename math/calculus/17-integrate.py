#!/usr/bin/env python3
"""Module that contains the function poly_integral"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(
            C, (int, float)):
        return None
    if poly == [0]:
        return [C]

    integral = [C]
    for i, coef in enumerate(poly):
        val = coef / (i + 1)
        if val.is_integer():
            val = int(val)
        integral.append(val)

    return integral
