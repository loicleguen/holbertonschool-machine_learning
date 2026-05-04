#!/usr/bin/env python3
"""Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable using the Adam optimization algorithm"""
    v_new = beta1 * v + (1 - beta1) * grad
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    v_corr = v_new / (1 - beta1 ** t)
    s_corr = s_new / (1 - beta2 ** t)
    var_new = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)
    return var_new, v_new, s_new
