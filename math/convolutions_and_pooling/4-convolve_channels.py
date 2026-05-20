#!/usr/bin/env python3
"""write a function that performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing multiple
                images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
            c: number of channels in the image
        kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel for
                the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding: is either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple:
                ph: padding for the height of the image
                pw: padding for the width of the image
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
    )

    conv_h = (h + 2 * ph - kh) // sh + 1
    conv_w = (w + 2 * pw - kw) // sw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] *
                kernel, axis=(1, 2, 3)
            )

    return convolved_images
