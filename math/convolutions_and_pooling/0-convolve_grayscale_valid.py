#!/usr/bin/env python3
"""Module for valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel

    Returns:
        numpy.ndarray containing the convolved images with shape
        (m, h - kh + 1, w - kw + 1)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    out_h = h - kh + 1
    out_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, out_h, out_w))

    # Perform convolution with only 2 for loops
    for i in range(out_h):
        for j in range(out_w):
            # Extract the patch from all images at position (i, j)
            patch = images[:, i:i+kh, j:j+kw]
            # Perform element-wise multiplication and sum
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
