#!/usr/bin/env python3
"""Scatter plot of height and weight for 2000 individuals"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Scatter plot of height and weight for 2000 individuals"""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, c='m', s=10)
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")
    plt.show()
