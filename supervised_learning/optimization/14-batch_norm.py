#!/usr/bin/env python3
"""creates a batch normalization layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )
    z = dense(prev)

    # Batch normalization parameters
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Compute mean and variance across the batch
    mean, variance = tf.nn.moments(z, axes=[0])

    # Apply batch normalization
    epsilon = 1e-7
    z_norm = tf.nn.batch_normalization(
        z,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    # Apply activation function
    return activation(z_norm)
