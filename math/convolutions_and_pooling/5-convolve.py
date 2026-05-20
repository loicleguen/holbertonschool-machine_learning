#!/usr/bin/env python3
"""write a function that performs a convolution
on images using multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernels

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernels: numpy.ndarray with shape (kh, kw, c, nc)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple of (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    conv_h = (h + 2 * ph - kh) // sh + 1
    conv_w = (w + 2 * pw - kw) // sw + 1
    convolved_images = np.zeros((m, conv_h, conv_w, nc))

    for i in range(conv_h):
        for j in range(conv_w):
            for k in range(nc):
                convolved_images[:, i, j, k] = np.sum(
                    padded_images[
                        :, i * sh:i * sh + kh, j * sw:j * sw + kw, :
                    ] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return convolved_images
