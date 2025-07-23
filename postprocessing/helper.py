import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
import shlex

# Gaussian kernel and its derivatives
def kernel_function(x, mu):
    return (1 / (np.sqrt(2 * np.pi) * mu)) * np.exp(-x**2 / (2 * mu**2))

def kernel_derivative(x, mu):
    return -x / (mu**2) * kernel_function(x, mu)

def kernel_second_derivative(x, mu):
    return (x**2 / mu**4 - 1 / mu**2) * kernel_function(x, mu)

# Quotient rule helper
def quotient_rule(nominator, denominator, dN, dD):
    return (dN * denominator- nominator* dD) / (denominator ** 2)

def find_all_roots(f, x_min, x_max, steps=1000):
    x = np.linspace(x_min, x_max, steps)
    roots = []
    for i in range(steps - 1):
        if f(x[i]) * f(x[i + 1]) < 0:  # Sign change
            try:
                root = brentq(f, x[i], x[i + 1])
                roots.append(root)
            except ValueError:
                pass
    return roots