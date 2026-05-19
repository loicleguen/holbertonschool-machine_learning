#!/usr/bin/env python3
"""Convolution with same padding on grayscale images."""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    images: numpy.ndarray of shape (m, h, w)
    kernel: numpy.ndarray of shape (kh, kw)
    Retourne: numpy.ndarray of shape (m, h, w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_top = (kh - 1) // 2
    pad_bottom = (kh - 1) - pad_top
    pad_left = (kw - 1) // 2
    pad_right = (kw - 1) - pad_left

    images_padded = np.pad(images,
                           ((0, 0),
                            (pad_top, pad_bottom),
                            (pad_left, pad_right)
                            ),
                           mode='constant', constant_values=0)

    out_h, out_w = h, w
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = images_padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
