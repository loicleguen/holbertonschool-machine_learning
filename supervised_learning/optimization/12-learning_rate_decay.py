#!/usr/bin/env python3
"""12-learning_rate_decay.py"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """creates a learning rate decay operation in
    tensorflow using inverse time decay"""
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_rate=decay_rate,
        decay_steps=decay_step,
        staircase=True
    )
