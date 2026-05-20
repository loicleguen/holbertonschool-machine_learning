#!/usr/bin/env python3
"""write a function that performs pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images

    Args:
        images: numpy.ndarray with shape (m, h, w, c)
        kernel_shape: tuple of (kh, kw)
        stride: tuple of (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1
    pooled_images = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(patch, axis=(1, 2))
            else:
                pooled_images[:, i, j, :] = np.mean(patch, axis=(1, 2))

    return pooled_images
