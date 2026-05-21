#!/usr/bin/env python3
"""Convolutional forward propagation."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer.

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b: numpy.ndarray of shape (1, 1, 1, c_new)
    activation: activation function to apply
    padding: 'same' or 'valid'
    stride: tuple (sh, sw)
    Returns: numpy.ndarray of shape (m, h_output, w_output, c_new)"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        pad_h = 0
        pad_w = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    h_output = (h_prev + 2 * pad_h - kh) // sh + 1
    w_output = (w_prev + 2 * pad_w - kw) // sw + 1

    A_padded = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )

    convolved = np.zeros((m, h_output, w_output, c_new))

    for i in range(h_output):
        for j in range(w_output):
            region = A_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            for k in range(c_new):
                convolved[:, i, j, k] = np.sum(region * W[:, :, :, k],
                                               axis=(1, 2, 3))

    return activation(convolved + b)
