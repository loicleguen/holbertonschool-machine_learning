#!/usr/bin/env python3
"""creates the training operation for a neural
network in tensorflow using momentum optimization"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """creates the training operation for a neural
    network in tensorflow using"""
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
