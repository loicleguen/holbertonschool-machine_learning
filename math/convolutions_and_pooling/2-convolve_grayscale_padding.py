#!/usr/bin/env python3
"""write a function that performs a convolution
on grayscale images with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
        padding: tuple of (ph, pw)
            ph: padding for the height
            pw: padding for the width

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    out_h = h + (2 * ph) - kh + 1
    out_w = w + (2 * pw) - kw + 1
    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return convolved_images
