import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
import shlex
from scipy.optimize import brentq
import pandas as pd

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

        num = d2N * denominator**2 - 2 * dN * denominator* dD + 2 * nominator* dD**2 - nominator* d2D * denominator
        denom = denominator**3
        d2y_pred.append(num / denom)
    return np.array(d2y_pred)

def plot_file(runs_dir, name="MoveProbgradU", save_dir="analysis"):
    folders = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
    for folder in folders:
        files = glob.glob(os.path.join(folder, f"{name}_*.dat"))
        if files:
            data_file = files[0]
            break
    if data_file:
        arr = np.loadtxt(data_file)
        plt.figure()
        plt.plot(arr)
        plt.title(f"Data from {data_file}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.savefig(os.path.join(save_dir, f"{name}_plot.png"))
        plt.close()
        print(f"MoveProb map plot saved to {os.path.join(save_dir, f'{name}_plot.png')}")
    else:
        print(f"No {name}_*.dat file found for plotting.")

def plot_csv(file_path, save_dir="analysis"):
    # Load the data from CSV
    df = pd.read_csv(file_path, sep='\t')

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Extract filename without extension for plot naming
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Determine plot type by columns available
    if {'nr_simulation', 'gamma', 'lambda'}.issubset(df.columns):
        # Plot for the gamma_lambda_results_constant.csv type
        plt.figure(figsize=(8,5))
        plt.plot(df['nr_simulation'], df['gamma'].astype(float), label='Gamma')
        plt.plot(df['nr_simulation'], df['lambda'].astype(float), label='Lambda')
        plt.xlabel('Simulation Number')
        plt.ylabel('Value')
        plt.title('Gamma and Lambda over Simulations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_plot = os.path.join(save_dir, f"{base_name}_plot.png")
        plt.savefig(output_plot)
        plt.close()
        print(f"Plot saved to {output_plot}")

    elif {'nr_slice', 'gamma', 'lambda'}.issubset(df.columns):
        # Plot for the gamma_lambda_results_densitydep.csv type
        plt.figure(figsize=(10,6))
        plt.plot(df['nr_slice'], df['gamma'].astype(float), label='Gamma')
        plt.plot(df['nr_slice'], df['lambda'].astype(float), label='Lambda')
        plt.xlabel('Slice Number')
        plt.ylabel('Values')
        plt.title('Density-dependent Gamma and Lambda')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_plot = os.path.join(save_dir, f"{base_name}_plot.png")
        plt.savefig(output_plot)
        plt.close()
        print(f"Plot saved to {output_plot}")

    else:
        print("CSV format not recognized for plotting.")


