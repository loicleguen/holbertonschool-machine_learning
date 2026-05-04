#!/usr/bin/env python3
"""calculates the learning rate decay"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """calculates the learning rate decay"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
