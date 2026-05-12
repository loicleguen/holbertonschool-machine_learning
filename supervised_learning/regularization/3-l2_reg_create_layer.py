#!/usr/bin/env python3
"""3-l2_reg_create_layer.py"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization"""
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha))
    return layer(prev)
