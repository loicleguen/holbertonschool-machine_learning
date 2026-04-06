#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Draw a line plot of y = x^3 for x in the range [0, 10]."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r')
    plt.xlim(0, 10)
    plt.show()
