#!/usr/bin/env python3
"""write a function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_top = (kh - 1) // 2
    pad_bottom = kh // 2
    pad_left = (kw - 1) // 2
    pad_right = kw // 2

    padded_images = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            patch = padded_images[:, i:i + kh, j:j + kw]
            convolved_images[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return convolved_images
