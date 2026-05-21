#!/usr/bin/env python3
"""Pooling forward propagation."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer.

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'
    Returns: numpy.ndarray of shape (m, h_output, w_output, c_prev)"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_output = int((h_prev - kh) // sh + 1)
    w_output = int((w_prev - kw) // sw + 1)

    output = np.zeros((m, h_output, w_output, c_prev))

    for i in range(h_output):
        for j in range(w_output):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            region = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
