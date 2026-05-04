#!/usr/bin/env python3
"""Update a variable using the Momentum optimization algorithm"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable using the Momentum optimization algorithm"""
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new
    return var_new, v_new
