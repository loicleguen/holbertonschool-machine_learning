#!/usr/bin/env python3
"""Convolution sur images en niveaux de gris avec padding et stride."""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray of shape (m, h, w)
        kernel: numpy.ndarray of shape (kh, kw)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    ph = padded_images.shape[1]
    pw = padded_images.shape[2]
    out_h = ((ph - kh) // sh) + 1
    out_w = ((pw - kw) // sw) + 1

    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return convolved_images
