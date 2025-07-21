import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
import shlex

from postprocessing.helper import kernel_function, kernel_derivative, kernel_second_derivative, \
    quotient_rule

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

def print_single_heatmap(file_path=None, data=None, title=None, save_path=None, show=True):
    """
    Simple function to print a single heatmap from an occupancy file or data array
    
    Args:
        file_path: Path to the occupancy .dat file (either this or data must be provided)
        data: Numpy array of occupancy data (either this or file_path must be provided)
        title: Optional custom title for the plot
        save_path: Optional path to save the image (if None, only displays)
        show: Whether to display the plot (default True)
    """
    try:
        # Load data from file or use provided data
        if data is not None:
            plot_data = data
        elif file_path is not None:
            plot_data = np.loadtxt(file_path)
            # Handle both flattened (1D) and proper 2D format
            if plot_data.ndim == 1 and len(plot_data) == 4000:  # 100 * 40 = 4000
                plot_data = plot_data.reshape(40, 100)
        else:
            raise ValueError("Either file_path or data must be provided")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        max_val = int(np.max(plot_data))
        discrete_cmap = create_discrete_colormap(max_val)
        im = ax.imshow(plot_data, cmap=discrete_cmap, origin='lower', aspect='equal', 
                       vmin=0, vmax=max_val)
        
        if title:
            ax.set_title(title)
        else:
            # Extract info from filename if available
            if file_path:
                filename = os.path.basename(file_path)
                ax.set_title(f'Occupancy Heatmap: {filename}')
            else:
                ax.set_title('Occupancy Heatmap')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ticks = list(range(0, max_val + 1))  # Integer ticks from 0 to max
        cbar = plt.colorbar(im, ax=ax, label='Occupancy Level', ticks=ticks)
        cbar.set_ticklabels([str(i) for i in ticks])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()  # Close to save memory if not showing
        
    except Exception as e:
        print(f"Error creating heatmap: {e}")

def create_comparison_grid(results, save_dir=None, run_date="", number=None):
    """
    Create a grid comparison of all heatmaps
    """
    print("Creating comparison grid...")
    
    # Get unique densities and tumble rates
    densities = sorted(list(set(r['density'] for r in results)))
    tumble_rates = sorted(list(set(r['tumble_rate'] for r in results)))
    
    n_rows = len(densities)  # Each row = one density
    n_cols = len(tumble_rates)  # Each column = one tumble rate
    
    print(f"Grid layout: {n_rows} densities × {n_cols} tumble rates")
    print(f"Densities: {densities}")
    print(f"Tumble rates: {tumble_rates}")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different return types from plt.subplots
    if n_rows == 1 and n_cols == 1:
        # Single subplot: axes is a single Axes object
        axes = np.array([[axes]])
    elif n_rows == 1:
        # Single row: axes is a 1D array
        axes = np.array([axes])
    elif n_cols == 1:
        # Single column: axes is a 1D array
        axes = np.array([[ax] for ax in axes])
    # else: axes is already a 2D array
    
    # Find global min/max for consistent color scaling - discrete integer occupancy values
    all_data = [r['occupancy_data'] for r in results]
    global_min = 0  # Occupancy starts at 0
    global_max = int(np.max([np.max(data) for data in all_data]))  # Find actual maximum
    
    # Create discrete colormap
    discrete_cmap = create_discrete_colormap(global_max)
    
    # Create a dictionary for quick lookup of results by (density, tumble_rate)
    result_dict = {(r['density'], r['tumble_rate']): r for r in results}
    
    # Fill the grid organized by density (rows) and tumble rate (columns)
    for row, density in enumerate(densities):
        for col, tumble_rate in enumerate(tumble_rates):
            if (density, tumble_rate) in result_dict:
                result = result_dict[(density, tumble_rate)]
                data = result['occupancy_data']
                
                im = axes[row, col].imshow(data, cmap=discrete_cmap, origin='lower', 
                                          aspect='equal', vmin=global_min, vmax=global_max)
                #axes[row, col].set_title(rf"$\rho$={density:.2f}, $\alpha$={tumble_rate:.2f}", 
                #                        fontsize=10)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                
                # Add metrics as text
                mean_occ = result['mean_occupancy']
                std_occ = result['std_occupancy']
                axes[row, col].text(0.02, 0.98, rf'$\mu={mean_occ:.2f}\,\ \sigma={std_occ:.2f}$', 
                                   transform=axes[row, col].transAxes, 
                                   verticalalignment='top', fontsize=8, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # No data for this combination
                axes[row, col].text(0.5, 0.5, 'No Data', 
                                   transform=axes[row, col].transAxes, 
                                   horizontalalignment='center', verticalalignment='center',
                                   fontsize=12)
                #axes[row, col].set_xticks([])
                #axes[row, col].set_yticks([])

    
    # Add row and column labels
    for row, density in enumerate(densities):
        axes[row, 0].set_ylabel(rf"{density:.2f}", fontsize=14, fontweight='bold')
    
    for col, tumble_rate in enumerate(tumble_rates):
        axes[0, col].set_xlabel(rf"{tumble_rate:.3f}", fontsize=14, fontweight='bold')
        axes[0, col].xaxis.set_label_position('top')
    
    # Add a common colorbar with adjusted spacing for side labels
    left_margin = 0.05
    right_margin = 0.88
    top_margin = 0.88
    bottom_margin = 0.1
    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)
    
    # Calculate automatic positioning for labels based on margins
    density_x = left_margin * 0.2  # 20% into the left margin
    density_y = (top_margin + bottom_margin) / 2  # Center vertically in plot area
    
    tumble_x = (left_margin + right_margin) / 2  # Center horizontally in plot area
    tumble_y = top_margin + (1 - top_margin) * 0.3  # 30% into the top margin space
    
    # Add larger axis labels on the sides with automatic positioning
    fig.text(density_x, density_y, r'Density $\rho$', fontsize=20, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(tumble_x, tumble_y, r'Tumble Rate $\alpha$', fontsize=20, fontweight='bold', 
             ha='center', va='center')
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    ticks = list(range(global_min, global_max + 1))  # Integer ticks from min to max
    cbar = fig.colorbar(im, cax=cbar_ax, label='Occupancy Level', ticks=ticks)
    cbar.set_label('Occupancy Level', fontsize=16, fontweight='bold')
    cbar.set_ticklabels([str(i) for i in ticks])
    cbar.ax.tick_params(labelsize=14)

    # Create title with date information
    # Get the total time from the first result (assuming all have the same total time)
    first_result = next(iter(results), None)
    total_time = first_result['totaltime'] if first_result and first_result['totaltime'] is not None else "Unknown"
    gamma = first_result['gamma'] if first_result and first_result['gamma'] is not None else None
    g = first_result['g'] if first_result and first_result['g'] is not None else None
    
    # Build parameter string for title
    param_str = ""
    
    # Extract potential type from run directory name
    potential_type = None
    if "uneven-sin" in run_date:
        potential_type = "Potential 1: Uneven sinusoidal move probability"
    elif "director-based-sin" in run_date:
        potential_type = "Potential 2: Director based move probability (uneven sinus)"
    elif "default" in run_date:
        potential_type = "No potential (move probability = 1)"
    
    # Add potential type to parameter string
    if potential_type:
        param_str += f", {potential_type}"
    
    if gamma is not None:
        param_str += f", γ={gamma}"
    if g is not None:
        param_str += f", g={g}"
    
    # Add title as text on the figure for better control
    plt.figtext(0.5, 0.98, 'Parameter Sweep Comparison Grid', 
                fontsize=40, fontweight='bold', ha='center')
    if run_date:
        plt.figtext(0.5, 0.94, f'Time: {total_time} steps, Run Date: {run_date[:8]}, Depicted Step: {number if number is not None else "Unknown"}{param_str}', 
                fontsize=18, ha='center')
    else:
        plt.figtext(0.5, 0.94, f'Time: {total_time} steps, Depicted Step: {number if number is not None else "Unknown"}{param_str}', 
                    fontsize=18, ha='center')
    
    folder = result['folder']
    folder_name = os.path.basename(folder)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_file = f'{save_dir}/comparison_grid_acc_{folder_name}.png'
    else:
        output_file = f'{folder}/comparison_grid_acc_.png'
    
    print(f"Saving comparison grid to '{output_file}'")

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to '{output_file}'")
    # plt.show()  # Comment out to only save, not display

def create_time_evolution_grid(all_data, time_files, dir_name, density, tumble_rate, save_dir=None):
    """Create a grid showing multiple time steps for evolution comparison"""
    
    # Select representative time points (max 9 for a 3x3 grid)
    n_points = min(9, len(all_data))
    indices = np.linspace(0, len(all_data)-1, n_points, dtype=int)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_points + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle single row or column cases
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Find global min/max for consistent color scaling - discrete integer occupancy values
    global_min = 0  # Occupancy starts at 0
    global_max = int(np.max([np.max(data) for data in all_data]))  # Find actual maximum
    
    # Create discrete colormap
    discrete_cmap = create_discrete_colormap(global_max)
    
    # Plot each selected time point
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        
        time_step, _ = time_files[idx]
        data = all_data[idx]
        
        im = axes[row, col].imshow(data, cmap=discrete_cmap, origin='lower', 
                                  aspect='equal', vmin=global_min, vmax=global_max)
        axes[row, col].set_title(f"Step {time_step}", fontsize=12)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_points, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    ticks = list(range(global_min, global_max + 1))  # Integer ticks from min to max
    cbar = fig.colorbar(im, cax=cbar_ax, label='Occupancy Level', ticks=ticks)
    cbar.set_ticklabels([str(i) for i in ticks])
    
    # Add title
    if density is not None and tumble_rate is not None:
        title = f"Time Evolution: Density={density:.2f}, Tumble Rate={tumble_rate:.3f}"
    else:
        title = f"Time Evolution: {dir_name}"
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save if requested
    if save_dir:
        filename = f"{save_dir}/evolution_grid.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Evolution grid saved to: {filename}")
    else:
        plt.show()
    
    plt.close()

def create_individual_movement_plot(timesteps, moving_counts, name, density, tumble_rate, gamma, g, run_date, save_dir=None):
    """Create an individual plot for one simulation's movement data"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use line plot without markers for better performance with many points
    if len(timesteps) > 100:
        ax.plot(timesteps, moving_counts, 'b-', linewidth=1.5)
    else:
        ax.plot(timesteps, moving_counts, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Number of Moving Particles', fontsize=12)
    
    # Create detailed title
    title = f'Moving Particles Over Time\n'
    title += f'Density={density:.3f}, Tumble Rate={tumble_rate:.3f}'
    if gamma is not None:
        title += f', γ={gamma:.3f}'
    if g is not None:
        title += f', G={g:.3f}'
    
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics as text
    mean_val = np.mean(moving_counts)
    std_val = np.std(moving_counts)
    ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        analysis_dir = f"{save_dir}/individual_movement_{name}.png"
    else:
        analysis_dir = f"analysis/individual_movement_{name}.png"
    # Save plot with optimized settings
    plt.savefig(analysis_dir, dpi=150, bbox_inches='tight', facecolor='white')  # Reduced DPI for smaller files
    plt.close()

def create_combined_movement_plots(all_data, run_date="", save_dir=None):
    """Create combined plots showing all simulations together"""
    
    # Plot 1: All trajectories on one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, data in enumerate(all_data):
        label = f'ρ={data["density"]:.2f}, α={data["tumble_rate"]:.3f}'
        ax.plot(data['timesteps'], data['moving_counts'], 
               linewidth=2, marker='o', markersize=3, label=label, alpha=0.8)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Number of Moving Particles', fontsize=12)
    ax.set_title('Movement Comparison: All Simulations', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        analysis_dir = f"{save_dir}/combined_movement_plot.png"
    else:
        analysis_dir = 'analysis/combined_movement_plot.png'

    plt.savefig(analysis_dir, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined trajectories plot saved: {analysis_dir}")
    
    # Plot 2: Average movement vs density (if we have multiple densities)
    densities = sorted(list(set(d['density'] for d in all_data)))
    tumble_rates = sorted(list(set(d['tumble_rate'] for d in all_data)))
    
    if len(tumble_rates) > 1:
        create_movement_plots_by_tumble_rate(all_data, run_date, save_dir=save_dir)

def create_movement_plots_by_tumble_rate(all_data, run_date="", save_dir=None):
    """Create separate plots for each tumble rate, showing different densities"""
    
    # Get unique tumble rates
    tumble_rates = sorted(list(set(d['tumble_rate'] for d in all_data)))
    
    for tumble_rate in tumble_rates:
        # Filter data for this tumble rate
        tumble_data = [d for d in all_data if abs(d['tumble_rate'] - tumble_rate) < 1e-6]
        
        if not tumble_data:
            continue
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot each density for this tumble rate
        for data in tumble_data:
            label = f'ρ={data["density"]:.3f}'
            # Use different markers and colors for different densities
            ax.plot(data['timesteps'], data['moving_counts'], 
                   linewidth=2, marker='o', markersize=3, label=label, alpha=0.8)
        
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Number of Moving Particles', fontsize=12)
        ax.set_title(f'Movement Over Time: Tumble Rate α={tumble_rate:.3f}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            analysis_dir = f"{save_dir}/combined_movement_plot_t_{tumble_rate}.png"
        else:
            analysis_dir = f"analysis/combined_movement_plot_t_{tumble_rate}.png"

        plt.savefig(analysis_dir, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Movement plot for tumble rate {tumble_rate:.3f} saved: {save_dir}")

def create_density_evolution_comparison_grid(all_processed_data, save_dir=None, run_date=None):
    """
    Create a comparison grid showing 2D stacked density evolution for all parameter combinations
    
    Args:
        all_processed_data: List of dictionaries containing processed density data for each parameter combination
        save_dir: Optional directory to save the comparison grid
    """
    if not all_processed_data:
        print("No processed data available for comparison grid")
        return

    # Get unique parameter values
    densities = sorted(list(set(item['density'] for item in all_processed_data)))
    tumble_rates = sorted(list(set(item['tumble_rate'] for item in all_processed_data)))

    n_rows = len(densities)
    n_cols = len(tumble_rates)

    print(f"Creating density evolution comparison grid: {n_rows} densities × {n_cols} tumble rates")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 3.5*n_rows))

    # Handle different subplot arrangements
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Find global min/max for consistent color scaling across all data
    all_density_2d = [item['density_2d'] for item in all_processed_data]
    global_min = np.min([np.min(data) for data in all_density_2d])
    global_max = np.max([np.max(data) for data in all_density_2d])

    # Create lookup dictionary for quick access
    data_dict = {(item['density'], item['tumble_rate']): item for item in all_processed_data}

    # Fill the grid
    for row, density in enumerate(densities):
        for col, tumble_rate in enumerate(tumble_rates):
            ax = axes[row, col]
            if (density, tumble_rate) in data_dict:
                item = data_dict[(density, tumble_rate)]
                density_2d = item['density_2d']
                time_steps = item['time_steps']

                # Create the heatmap
                im = ax.imshow(density_2d, cmap='viridis', aspect='auto',
                              origin='lower', interpolation='nearest',
                              vmin=global_min, vmax=global_max)

                # Set up y-tick labels to show time steps (subsample if too many)
                n_ticks = min(5, len(time_steps))
                tick_indices = np.linspace(0, len(time_steps)-1, n_ticks, dtype=int)
                ax.set_yticks(tick_indices)
                ax.set_yticklabels([str(time_steps[i]) for i in tick_indices], fontsize=8)

                # Remove x-ticks for cleaner look
                ax.set_xticks([])

                # Add mean and std as text
                mean_val = np.mean(density_2d)
                std_val = np.std(density_2d)
                ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                        transform=ax.transAxes, verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                # No data for this parameter combination
                ax.text(0.5, 0.5, 'No Data',
                        transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

    # Add row and column labels (left and top)
    for row, density in enumerate(densities):
        axes[row, 0].set_ylabel(f"{density:.2f}", fontsize=14, fontweight='bold')

    for col, tumble_rate in enumerate(tumble_rates):
        axes[0, col].set_xlabel(f"{tumble_rate:.3f}", fontsize=14, fontweight='bold')
        axes[0, col].xaxis.set_label_position('top')

    # Add larger axis labels on the sides/top
    left_margin = 0.08
    right_margin = 0.88
    top_margin = 0.88
    bottom_margin = 0.1
    fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

    # Calculate automatic positioning for labels based on margins
    density_x = left_margin * 0.2  # 20% into the left margin
    density_y = (top_margin + bottom_margin) / 2  # Center vertically in plot area
    tumble_x = (left_margin + right_margin) / 2  # Center horizontally in plot area
    tumble_y = top_margin + (1 - top_margin) * 0.3  # 30% into the top margin space

    # Add larger axis labels on the sides with automatic positioning
    fig.text(density_x, density_y, r'Density $\rho$', fontsize=20, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(tumble_x, tumble_y, r'Tumble Rate $\alpha$', fontsize=20, fontweight='bold',
             ha='center', va='center')

    # Add colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Density Value')
    cbar.set_label('Density Value', fontsize=16, fontweight='bold')

    # Add main title and subtitle
    plt.figtext(0.5, 0.98, 'Density Evolution Comparison Grid (2D Stacked)',
                fontsize=28, fontweight='bold', ha='center')
    plt.figtext(0.5, 0.95, 'Time evolution shown as stacked 1D profiles (vertical: time, horizontal: space)',
                fontsize=16, ha='center', style='italic')

    # Save the comparison grid
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if run_date:
            save_path = os.path.join(save_dir, f'density_evolution_comparison_grid_{run_date}.png')
        else:
            save_path = os.path.join(save_dir, 'density_evolution_comparison_grid.png')
    else:
        if run_date:
            save_path = f'analysis/density_evolution_comparison_grid_{run_date}.png'
        else:
            save_path = 'analysis/density_evolution_comparison_grid.png'
        os.makedirs('analysis', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Density evolution comparison grid saved to: {save_path}")
    plt.close()

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

def compute_density_derivatives(profile, mu=None, smooth=True, method=None):
    x = np.arange(len(profile))
    if mu is None:
        mu = np.std(profile)
    if smooth:
        smoothed = nw_kernel_regression(profile, x, x, mu)
    else:
        smoothed = profile

    if method == "kernel":
        first_deriv = nw_first_derivative(x, x, smoothed, mu)
        second_deriv = nw_second_derivative(x, x, smoothed, mu)
    elif method == "diff":
        first_deriv = np.gradient(smoothed)
        second_deriv = np.gradient(first_deriv)
    else:
        raise ValueError(f"Unknown method '{method}' for derivative computation, try diff or kernel")

    return smoothed, first_deriv, second_deriv

def compute_profiles_by_step(runs_dir, steps_to_include, smooth=True, mu=None, method=None):
    import bisect
    profiles_by_step = {}
    folders = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
    print(f"folders found: {folders}")
    # Pre-index all available density files per folder
    folder_step_map = {}
    for folder in folders:
        density_files = [f for f in os.listdir(folder) if f.startswith("Density_") and f.endswith(".dat")]
        steps = []
        file_map = {}
        for f in density_files:
            try:
                num = int(f.split("_")[1].split(".")[0])
                steps.append(num)
                file_map[num] = os.path.join(folder, f)
            except Exception:
                continue
        steps.sort()
        folder_step_map[folder] = (steps, file_map)
    # For each folder and each requested step, collect all files >= step and compute mean profile
    profiles_by_step = {}
    for folder in folders:
        steps, file_map = folder_step_map[folder]
        folder_name = os.path.basename(folder)
        for step in steps_to_include:
            profiles = []
            idx = bisect.bisect_left(steps, step)
            for s in steps[idx:]:
                path = file_map[s]
                if os.path.isfile(path):
                    data = np.loadtxt(path)
                    if data.ndim == 1:
                        profiles.append(data)
                    elif data.ndim == 2:
                        profiles.append(np.mean(data, axis=0))
            if not profiles:
                print(f"[Warning] No profiles found for folder {folder_name} step {step}")
                continue
            avg_profile = np.mean(profiles, axis=0)
            print(f"shape of average profile for {folder_name} step {step}: {avg_profile.shape}")
            smoothed, d1, d2 = compute_density_derivatives(avg_profile, mu=mu, smooth=smooth, method=method)
            # Key by (folder_name, step)
            profiles_by_step[(folder_name, step)] = (smoothed, d1, d2)
    return profiles_by_step

def plot_density_derivative_grid(profiles_by_step, save_choice=None, save_dir=None, title_prefix="Density & Derivatives", method=None):
    # Group keys by subfolder
    from collections import defaultdict
    folder_steps = defaultdict(list)
    for key in profiles_by_step:
        folder, step = key
        folder_steps[folder].append(step)
    for folder in folder_steps:
        steps = sorted(folder_steps[folder])
        n_rows = len(steps)
        n_cols = 3  # Profile, 1st Derivative, 2nd Derivative
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for i, step in enumerate(steps):
            profile, d1, d2 = profiles_by_step[(folder, step)]
            x = np.arange(len(profile))
            axes[i, 0].plot(x, profile)
            axes[i, 0].set_title(f"Smoothed Profile (Step {step}) for method '{method}'")
            axes[i, 0].set_ylabel("Density")
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 1].plot(x, d1, color='orange')
            axes[i, 1].set_title("1st Derivative")
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 2].plot(x, d2, color='green')
            axes[i, 2].set_title("2nd Derivative")
            axes[i, 2].grid(True, alpha=0.3)
        for ax in axes[-1]:
            ax.set_xlabel("X Position")
        plt.suptitle(f"{title_prefix} - {folder}", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_choice:
            save_path = f"{save_dir}/density_derivative_grid_{method}_{folder}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Density derivative grid saved to: {save_path}")
        else:
            plt.show()
