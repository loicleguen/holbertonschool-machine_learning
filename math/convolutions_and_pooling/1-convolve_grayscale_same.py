#!/usr/bin/env python3
"""write a function that performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
            kh: height of the kernel
            kw: width of the kernel
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')
    convolved_images = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))
    return convolved_images
