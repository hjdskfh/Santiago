import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import re
import seaborn as sns

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


# def plot_2d_family(df, value_col='lambda', title=None, save_path=None):
#     """
#     Plots value_col (lambda or gamma) vs rho for multiple amplitudes on a single 2D plot.
#     """
#     import seaborn as sns
#     sns.set(style='whitegrid')
    
#     df = df.dropna(subset=['amplitude', 'rho', value_col])
#     df['amplitude'] = df['amplitude'].astype(float)
#     df['rho'] = df['rho'].astype(float)
#     df[value_col] = df[value_col].astype(float)

#     plt.figure(figsize=(10, 6))
#     for amp in sorted(df['amplitude'].unique()):
#         subdf = df[df['amplitude'] == amp].sort_values('rho')
#         plt.plot(subdf['rho'], subdf[value_col], marker='o', label=f'Amp {amp:.2f}')
#     plt.xlabel('rho')
#     plt.ylabel(value_col)
#     plt.title(title or f'{value_col} vs rho')
#     plt.legend(title='Amplitude')
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#         print(f"[plot_2d_family] Saved plot to {save_path}")
#     # plt.show()

def plot_2d_family(df, value_col='lambda', title=None, save_path=None):
    """
    Plots value_col (lambda or gamma) vs rho for multiple amplitudes on a single 2D plot.
    Colors lines using a rainbow-like colormap by amplitude.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    df = df.dropna(subset=['amplitude', 'rho', value_col])
    df['amplitude'] = df['amplitude'].astype(float)
    df['rho'] = df['rho'].astype(float)
    df[value_col] = df[value_col].astype(float)

    amplitudes = sorted(df['amplitude'].unique())
    norm = mcolors.Normalize(vmin=min(amplitudes), vmax=max(amplitudes))
    cmap = cm.get_cmap('rainbow')  # You can try 'viridis', 'plasma', 'turbo', etc.

    plt.figure(figsize=(10, 6))
    for amp in amplitudes:
        subdf = df[df['amplitude'] == amp].sort_values('rho')
        color = cmap(norm(amp))
        plt.plot(subdf['rho'], subdf[value_col], marker='o', label=f'Amp {amp:.2f}', color=color)

    plt.xlabel('rho')
    plt.ylabel(value_col)
    plt.title(title or f'{value_col} vs rho')
    plt.legend(title='Amplitude', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[plot_2d_family] Saved plot to {save_path}")
    plt.show()

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

        # Plot lambda as 2D line plot colored by amplitude
        plot_2d_family(
            df,
            value_col='lambda',
            title=f'Lambda vs Rho (Colored by Amplitude) — {method}',
            save_path=os.path.join(analysis_base, f'lambda_vs_rho_lines_{method}_{result_type}.png')
        )

        # Plot gamma as 2D line plot colored by amplitude
        plot_2d_family(
            df,
            value_col='gamma',
            title=f'Gamma vs Rho (Colored by Amplitude) — {method}',
            save_path=os.path.join(analysis_base, f'gamma_vs_rho_lines_{method}_{result_type}.png')
        )
