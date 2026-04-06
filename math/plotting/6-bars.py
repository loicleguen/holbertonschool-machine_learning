#!/usr/bin/env python3
"""6. Bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar chart"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    fuits = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['r', 'y', '#ff8000', '#ffe5b4']

    bottom = np.zeros(3)
    for i in range(4):
        plt.bar(people, fruit[i], width=0.5, bottom=bottom, color=colors[i], label=fuits[i])
        bottom += fruit[i]

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend(loc='upper right')
    plt.show()
