#!/usr/bin/env python3
"""creates the training operation for a neural network"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm"""
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
