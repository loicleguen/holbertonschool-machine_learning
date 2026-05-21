#!/usr/bin/env python3
"""Pooling backward propagation."""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer.

    dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
    kernel_shape: tuple of (kh, kw)
    stride: tuple of (sh, sw)
    mode: 'max' or 'avg'
    Returns: dA_prev
    """
    m, h_prev, w_prev, c = A_prev.shape
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            for k in range(c):
                if mode == 'max':
                    a_slice = A_prev[:, vert_start:vert_end,
                                     horiz_start:horiz_end, k]
                    mask = a_slice == np.max(
                        a_slice, axis=(1, 2), keepdims=True)
                    dA_prev[:, vert_start:vert_end,
                            horiz_start:horiz_end, k] += (
                                mask * dA[:, i, j, k][:, np.newaxis,
                                                      np.newaxis]
                            )
                elif mode == 'avg':
                    da = dA[:, i, j, k][:, np.newaxis, np.newaxis]
                    average = da / (kh * kw)
                    dA_prev[:, vert_start:vert_end,
                            horiz_start:horiz_end, k] += average

    return dA_prev
