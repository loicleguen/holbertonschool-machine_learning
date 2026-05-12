#!/usr/bin/env python3
"""2-l2_reg_cost.py"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """calculates the cost of a neural network with L2 regularization"""
    l2_losses = model.losses
    return tf.stack([cost + l2_loss for l2_loss in l2_losses])
