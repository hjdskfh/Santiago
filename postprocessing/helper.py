import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
import shlex
from scipy.optimize import brentq

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

# Nadaraya-Watson regression
def nw_kernel_regression(x_eval, x_train, y_train, mu):
    y_pred = []
    for x in x_eval:
        dx = x - x_train
        weights = kernel_function(dx, mu)
        nominator = np.sum(weights * y_train)
        denominator = np.sum(weights)
        y_pred.append(nominator / denominator if denominator != 0 else 0)
    return np.array(y_pred)

# First derivative of regression
def nw_first_derivative(x_eval, x_train, y_train, mu):
    dy_pred = []
    for x in x_eval:
        dx = x - x_train
        weights = kernel_function(dx, mu)
        dw = kernel_derivative(dx, mu)
        nominator = np.sum(weights * y_train)
        denominator = np.sum(weights)
        dN = np.sum(dw * y_train)
        dD = np.sum(dw)
        dy_pred.append(quotient_rule(nominator, denominator, dN, dD))
    return np.array(dy_pred)

# Second derivative of regression
def nw_second_derivative(x_eval, x_train, y_train, mu):
    d2y_pred = []
    for x in x_eval:
        dx = x - x_train
        weights = kernel_function(dx, mu)
        dw = kernel_derivative(dx, mu)
        d2w = kernel_second_derivative(dx, mu)

        nominator= np.sum(weights * y_train)
        denominator= np.sum(weights)
        dN = np.sum(dw * y_train)
        dD = np.sum(dw)
        d2N = np.sum(d2w * y_train)
        d2D = np.sum(d2w)

        num = d2N * denominator**2 - 2 * dN * denominator* dD + 2 * nominator* dD**2 - nominator* d2D
        denom = denominator**3
        d2y_pred.append(num / denom)
    return np.array(d2y_pred)
