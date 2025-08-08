import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import re

from postprocessing.engine import extract_parameters_from_folder, extract_amplitude_from_log

def collect_results_for_3d_plot(results_dirs, method='diff', result_type='densitydep'):
    """
    Collects amplitude, rho, lambda, gamma from all result directories.
    Returns a DataFrame with columns: amplitude, rho, lambda, gamma
    """
    all_rows = []
    for result_dir in results_dirs:
        # Try to extract amplitude from folder name first
        dir_name = os.path.basename(result_dir)
        _, _, _, _, amplitude = extract_parameters_from_folder(dir_name)
        # If amplitude is not found, try to extract from run_summary.log in parent directory
        if amplitude is None:
            parent_dir = os.path.dirname(result_dir)
            log_path = os.path.join(parent_dir, 'run_summary.log')
            if os.path.exists(log_path):
                amplitude = extract_amplitude_from_log(log_path)
            else:
                print(f"No amplitude found in folder name or run_summary.log for {result_dir}")
                amplitude = None
        # Find the results file
        result_file = os.path.join(result_dir, f'gamma_lambda_results_{method}_{result_type}.csv')
        if not os.path.exists(result_file):
            print(f"Skipping {result_file} (not found)")
            continue
        df = pd.read_csv(result_file, sep='\t')
        for i, row in df.iterrows():
            all_rows.append({
                'amplitude': amplitude,
                'rho': row['rho'],
                'lambda': float(row['lambda']),
                'gamma': float(row['gamma'])
            })
            print(f"rho: {row['rho']}, amplitude: {amplitude}, lambda: {row['lambda']}, gamma: {row['gamma']}")
    return pd.DataFrame(all_rows)

def plot_3d_surface(df, value_col='lambda', title=None, save_path=None):
    """
    Plots a 3D surface of value_col (lambda or gamma) vs rho and amplitude.
    """
    # Clean and convert data to float
    df = df.dropna(subset=['amplitude', 'rho', value_col])
    df['amplitude'] = df['amplitude'].astype(float)
    df['rho'] = df['rho'].astype(float)
    df[value_col] = df[value_col].astype(float)

    print(f"[plot_3d_surface] DataFrame shape: {df.shape}")
    if df.empty:
        print(f"[plot_3d_surface] DataFrame is empty after cleaning. Skipping plot for {value_col}.")
        return

    pivot = df.pivot_table(index='amplitude', columns='rho', values=value_col)
    print(f"[plot_3d_surface] Pivot table shape: {pivot.shape}")
    if pivot.empty or np.all(np.isnan(pivot.values)):
        print(f"[plot_3d_surface] Pivot table is empty or all NaN. Skipping plot for {value_col}.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
    Z = pivot.values
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('rho')
    ax.set_ylabel('amplitude')
    ax.set_zlabel(value_col)
    # Also plot 2D slices: gamma/lambda vs rho for each amplitude
    for amp in sorted(df['amplitude'].unique()):
        subdf = df[df['amplitude'] == amp]
        if subdf.empty:
            print(f"[plot_3d_surface] No data for amplitude {amp}, skipping 2D plot.")
            continue
        plt.figure(figsize=(8,5))
        plt.plot(subdf['rho'], subdf[value_col], marker='o', label=f'Amplitude={amp}')
        plt.xlabel('rho')
        plt.ylabel(value_col)
        plt.title(f'{value_col.capitalize()} vs Rho (Amplitude={amp})')
        plt.legend()
        plt.grid(True)
        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_2D_amp{amp}{ext}", dpi=300, bbox_inches='tight')
            print(f"Saved 2D plot for amplitude {amp} to {base}_2D_amp{amp}{ext}")
        plt.close()
    if title:
        ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    analysis_base = "analysis"
    result_type = 'densitydep'  # or 'constant'
    methods = ['diff', 'kernel']  # List all methods you want to process

    for method in methods:
        pattern = os.path.join(analysis_base, '**', f'results_gamma_lambda_{method}_{result_type}_*')
        result_dirs = glob.glob(pattern, recursive=True)
        print(f"Found {len(result_dirs)} result directories for method '{method}'.")
        df = collect_results_for_3d_plot(result_dirs, method=method, result_type=result_type)
        print(f"Collected {len(df)} rows for method '{method}'.")
        # Plot lambda
        plot_3d_surface(df, value_col='lambda', title=f'Lambda vs Rho and Amplitude ({method})',
                        save_path=os.path.join(analysis_base, f'lambda_vs_rho_amplitude_{method}_{result_type}.png'))
        # Plot gamma
        plot_3d_surface(df, value_col='gamma', title=f'Gamma vs Rho and Amplitude ({method})',
                        save_path=os.path.join(analysis_base, f'gamma_vs_rho_amplitude_{method}_{result_type}.png'))