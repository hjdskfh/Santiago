import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
import shlex

def create_discrete_colormap(max_val):
    """
    Create a discrete colormap for occupancy values from 0 to max_val
    """
    # Use viridis colors but make them discrete
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0, 1, max_val + 1))
    return ListedColormap(colors)

def extract_parameters_from_folder(folder_name):
    """
    Extract density, tumbling rate, gamma, and g from folder name
    Assumes folder naming convention like: d0.5_t0.1_time10000_gamma-0.5_g1
    """
    # Extract basic parameters
    density_match = re.search(r'd([0-9]*\.?[0-9]+)', folder_name)
    tumble_match = re.search(r't([0-9]*\.?[0-9]+)', folder_name)
    time_match = re.search(r'time([0-9]*\.?[0-9]+)', folder_name)
    
    # Extract gamma and g parameters (with support for negative values)
    gamma_match = re.search(r'gamma(-?[0-9]*\.?[0-9]+)', folder_name)
    g_match = re.search(r'g(-?[0-9]*\.?[0-9]+)', folder_name)
    
    if density_match and tumble_match:
        density = float(density_match.group(1))
        tumble_rate = float(tumble_match.group(1))
        total_time = float(time_match.group(1)) if time_match else None
        gamma = float(gamma_match.group(1)) if gamma_match else None
        g = float(g_match.group(1)) if g_match else None
        
        return density, tumble_rate, total_time, gamma, g
    
    return None, None, None, None, None


def calculate_metrics(occupancy_data):
    """
    Calculate useful metrics from occupancy data
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean_occupancy'] = np.mean(occupancy_data)
    metrics['max_occupancy'] = np.max(occupancy_data)
    metrics['std_occupancy'] = np.std(occupancy_data)
    
    # Spatial patterns
    metrics['empty_sites'] = np.sum(occupancy_data == 0) / occupancy_data.size
    metrics['full_sites'] = np.sum(occupancy_data >= 3) / occupancy_data.size
    
    return metrics


def load_occupancy_data(folder_path, time_step):
    """Load occupancy data for a specific time step, with fallback logic."""
    occupancy_files = glob.glob(f"{folder_path}/Occupancy_*.dat")
    if not occupancy_files:
        return None, None
    
    # Extract available time steps
    available_times = []
    for file in occupancy_files:
        match = re.search(r'Occupancy_(-?\d+)\.dat', file)
        if match:
            available_times.append(int(match.group(1)))
    
    if not available_times:
        return None, None
    
    available_times.sort()
    
    # Find best match
    if time_step in available_times:
        actual_time = time_step
    elif time_step == -1:
        actual_time = -1 if -1 in available_times else min(available_times)
    else:
        actual_time = min(available_times, key=lambda x: abs(x - time_step))
    
    # Load the data
    target_file = f"{folder_path}/Occupancy_{actual_time}.dat"
    try:
        data = np.loadtxt(target_file)
        # Handle both flattened (1D) and proper 2D format
        if data.ndim == 1 and len(data) == 4000:  # 100 * 40 = 4000
            # Reshape flattened data to 2D (40 rows, 100 cols)
            data = data.reshape(40, 100)
        return data, actual_time
    except Exception as e:
        print(f"Error loading {target_file}: {e}")
        return None, None

def process_folder_for_sweep(folder_path, time_step):
    """Process a single folder for parameter sweep analysis."""
    folder_name = os.path.basename(folder_path)
    density, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(folder_name)
    
    if density is None or tumble_rate is None:
        return None
    
    # Skip if time step is impossible
    if total_time and time_step > 0 and time_step > total_time:
        return None
    
    data, actual_time = load_occupancy_data(folder_path, time_step)
    if data is None:
        return None
    
    metrics = calculate_metrics(data)
    return {
        'folder': folder_path,
        'density': density,
        'tumble_rate': tumble_rate,
        'totaltime': total_time,
        'gamma': gamma,
        'g': g,
        'occupancy_data': data,
        **metrics
    }

def find_files_in_directory(directory, pattern="Occupancy_*.dat", prefix_filter=None):
    """
    Helper function to find files in directories with optional filtering
    
    Args:
        directory: Base directory to search
        pattern: File pattern to match
        prefix_filter: Optional prefix filter for subdirectories (e.g., "START_")
    
    Returns:
        List of tuples: (subdirectory_name, full_path_to_subdir, matching_files)
    """
    if not os.path.exists(directory):
        return []
    
    results = []
    for item in os.listdir(directory):
        # Apply prefix filter if specified
        if prefix_filter and not item.startswith(prefix_filter):
            continue
        
        subdir_path = os.path.join(directory, item)
        if os.path.isdir(subdir_path):
            matching_files = glob.glob(f"{subdir_path}/{pattern}")
            if matching_files:
                results.append((item, subdir_path, matching_files))
    
    return sorted(results)

# Which parameters to extract
PARAMS_TO_TRACK = ["--density", "--tumble-rate", "--gamma", "--g", "--potential-lower", "--potential-upper"]

def parse_cmd_line(cmd_line):
    args = shlex.split(cmd_line)
    param_dict = {}
    for i, arg in enumerate(args):
        if arg in PARAMS_TO_TRACK and i + 1 < len(args):
            param_dict[arg.lstrip("--")] = args[i + 1]
    return param_dict

def find_cmd_files(directory):
    return glob.glob(os.path.join(directory, "**/*.cmd"), recursive=True)

def load_cmds(directory):
    data = []
    for cmd_file in find_cmd_files(directory):
        with open(cmd_file, 'r') as f:
            lines = f.readlines()
            # Find the first non-comment line that looks like a command
            cmd_line = next((line for line in lines if line.strip().startswith('./')), None)
            if cmd_line:
                params = parse_cmd_line(cmd_line)
                params['filename'] = cmd_file
                data.append(params)
    return data

def plot_summary(data):
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), squeeze=False)
    for i, run in enumerate(data):
        ax = axes[0][i]
        # Example: plot gamma vs density
        density = float(run.get("density", 0))
        gamma = float(run.get("gamma", 0))
        ax.plot(gamma, density, 'ro')
        ax.set_title(f'Run {i+1}\nGamma={gamma}, Density={density}')
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Density')
    plt.tight_layout()
    plt.show()

"""# Example usage
if __name__ == "__main__":
    cmd_data = load_cmds("runs/")
    plot_summary(cmd_data)
"""