#!/usr/bin/env python3

def moving_average(data, beta):
    """Calculates the moving average of a data set"""
    v = 0
    m_avg = []
    for t, x in enumerate(data, 1):
        v = beta * v + (1 - beta) * x
        v_corr = v / (1 - beta ** t)
        m_avg.append(v_corr)
    return m_avg
