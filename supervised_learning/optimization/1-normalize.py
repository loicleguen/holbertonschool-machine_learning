#!/usr/bin/env python3
"""Normalize a matrix"""
import numpy as np


def normalize(X, m, s):
    """Normalize a matrix"""
    return (X - m) / s
