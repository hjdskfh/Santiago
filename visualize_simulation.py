import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime

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

def create_individual_heatmaps(results):
    """
    Create individual heatmap files for each parameter combination
    """
    print("Creating individual heatmaps...")
    
    for result in results:
        folder = result['folder']
        density = result['density']
        tumble_rate = result['tumble_rate']
        totaltime = result['totaltime']
        data = result['occupancy_data']
        
        # Create title and save path
        title = f'Density={density:.2f}, Tumbling Rate={tumble_rate:.3f}, Time={totaltime}'
        folder_name = os.path.basename(folder)
        output_file = f'{folder}/heatmap_{folder_name}.png'
        
        # Use print_single_heatmap but with data instead of file path
        print_single_heatmap(data=data, title=title, save_path=output_file, show=False)
        
    print(f"Created {len(results)} individual heatmaps")

def create_comparison_grid(results, run_date="", number=None):
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
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
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
        plt.figtext(0.5, 0.94, f'Time: {total_time} steps, Run Date: {run_date[:9]}, Depicted Step: {number if number is not None else "Unknown"}{param_str}', 
                fontsize=18, ha='center')
    else:
        plt.figtext(0.5, 0.94, f'Time: {total_time} steps, Depicted Step: {number if number is not None else "Unknown"}{param_str}', 
                    fontsize=18, ha='center')
    
    # Create filename with date information
    if run_date:
        filename = f'analysis/comp_{run_date}__{number if number is not None else "unknown"}.png'
    else:
        filename = f'analysis/comp_{number if number is not None else "unknown"}.png'
    
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to '{filename}'")
    # plt.show()  # Comment out to only save, not display

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
        return data, actual_time
    except:
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

def create_parameter_sweep_visualization(runs_dir='runs', number=1, process_all_times=False):
    """Create comprehensive visualizations for parameter sweep results."""
    os.makedirs('analysis', exist_ok=True)
    
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Extract run date
    run_date = ""
    if "run_" in runs_dir:
        timestamp = os.path.basename(runs_dir)[4:]  # Remove "run_" prefix
        try:
            run_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S")
        except:
            run_date = timestamp.replace(":", "").replace(" ", "_")
    
    # Get folders with occupancy files
    folder_results = find_files_in_directory(runs_dir)
    folders = [full_path for _, full_path, _ in folder_results]
    
    if not folders:
        print(f"No simulation folders found in '{runs_dir}'!")
        return
    
    # Process all time steps if requested
    if process_all_times:
        all_times = set()
        for _, _, files in folder_results:
            for file in files:
                match = re.search(r'Occupancy_(-?\d+)\.dat', file)
                if match:
                    all_times.add(int(match.group(1)))
        
        if not all_times:
            print("No valid occupancy files found!")
            return
        
        for time_step in sorted(all_times):
            print(f"Processing time step {time_step}...")
            create_parameter_sweep_visualization(runs_dir, number=time_step, process_all_times=False)
        return
    
    # Process single time step
    results = []
    for folder in folders:
        result = process_folder_for_sweep(folder, number)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    print(f"Processing {len(results)} results for time step {number}")
    create_individual_heatmaps(results)
    create_comparison_grid(results, run_date, number)

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

def print_multiple_heatmaps(runs_dir, time_step=-1, save_dir=None, prefix_filter=None, show_individual=True):
    """Print heatmaps for multiple directories."""
    results = find_files_in_directory(runs_dir, prefix_filter=prefix_filter)
    
    if not results:
        filter_msg = f" with prefix '{prefix_filter}'" if prefix_filter else ""
        print(f"No directories{filter_msg} found in {runs_dir}")
        return
    
    print(f"Found {len(results)} directories to process")
    
    for subdir_name, full_path, _ in results:
        data, actual_time = load_occupancy_data(full_path, time_step)
        if data is None:
            continue
            
        # Create title based on directory type
        if prefix_filter == "START_":
            density_match = re.search(r'd([0-9]*\.?[0-9]+)', subdir_name)
            density = float(density_match.group(1)) if density_match else "Unknown"
            title = f"START Configuration: Density={density}, Step={actual_time}"
        else:
            title = f"{subdir_name} - Step {actual_time}"
        
        # Set up save path
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{subdir_name}_step{actual_time}.png"
        
        print_single_heatmap(data=data, title=title, save_path=save_path, show=show_individual)

def visualize_time_evolution(directory_path, save_dir=None, show_individual=False):
    """
    Visualize the time evolution of a single configuration directory
    
    Args:
        directory_path: Path to directory containing time series occupancy files
        save_dir: Optional directory to save images and animation
        show_individual: Whether to display each time step individually
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Find all occupancy files and sort by time step
    occupancy_files = glob.glob(f"{directory_path}/Occupancy_*.dat")
    if not occupancy_files:
        print(f"No occupancy files found in {directory_path}")
        return
    
    # Extract time steps and sort
    time_files = []
    for file in occupancy_files:
        match = re.search(r'Occupancy_(-?\d+)\.dat', file)
        if match:
            time_step = int(match.group(1))
            time_files.append((time_step, file))
    
    time_files.sort()  # Sort by time step
    
    if len(time_files) < 2:
        print(f"Need at least 2 time steps for evolution visualization, found {len(time_files)}")
        return
    
    print(f"Found {len(time_files)} time steps from {time_files[0][0]} to {time_files[-1][0]}")
    
    # Extract parameters from directory name
    dir_name = os.path.basename(directory_path)
    density, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
    
    # Create output directory if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving to: {save_dir}")
    
    # Create individual frames
    all_data = []
    for i, (time_step, file_path) in enumerate(time_files):
        try:
            data = np.loadtxt(file_path)
            all_data.append(data)
            
            if show_individual or save_dir:
                title = f"Time Evolution: {dir_name}\nStep {time_step}"
                if density is not None and tumble_rate is not None:
                    title = f"Density={density:.2f}, Tumble Rate={tumble_rate:.3f}\nTime Step: {time_step}"
                
                save_path = None
                if save_dir:
                    save_path = f"{save_dir}/step_{time_step:06d}.png"
                
                print_single_heatmap(data=data, title=title, save_path=save_path, show=show_individual)
                
        except Exception as e:
            print(f"Error processing time step {time_step}: {e}")
            continue
    
    # Create comparison grid showing evolution
    if len(all_data) >= 4:  # Only create grid if we have enough frames
        create_evolution_grid(all_data, time_files, dir_name, density, tumble_rate, save_dir)
    
    print(f"Time evolution visualization complete! Processed {len(all_data)} time steps.")

def create_evolution_grid(all_data, time_files, dir_name, density, tumble_rate, save_dir=None):
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

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments and see which runs directory to use
    if len(sys.argv) > 1:
        runs_dir = sys.argv[1]
        print(f"Using runs directory: {runs_dir}")
    else:
        # Default behavior - look for the most recent run
        runs_base = "runs"
        if os.path.exists(runs_base):
            # Find the most recent timestamped run directory
            run_dirs = [d for d in os.listdir(runs_base) if d.startswith("run_") and os.path.isdir(os.path.join(runs_base, d))]
            if run_dirs:
                # Sort by timestamp and take the most recent
                run_dirs.sort()
                runs_dir = os.path.join(runs_base, run_dirs[-1])
                print(f"Using most recent run directory: {runs_dir}")
            else:
                runs_dir = runs_base
                print(f"No timestamped runs found, using: {runs_dir}")
        else:
            runs_dir = "runs"
            print(f"Using default runs directory: {runs_dir}")
    
    print("Starting simulation visualization...")
    
    # Show options to user
    print("\nChoose visualization mode:")
    print("1. Parameter sweep comparison grid (specific time step)")
    print("2. Parameter sweep grids for ALL time steps")
    print("3. View START configuration heatmaps")
    print("4. View all heatmaps in directory") 
    print("5. View single heatmap from file path")
    print("6. View time evolution of single configuration")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1-6): ").strip()
            if mode_choice in ['1', '2', '3', '4', '5', '6']:
                break
            else:
                print("Please enter 1, 2, 3, 4, 5, or 6.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
    
    if mode_choice == '1':
        # Parameter sweep grid for specific time step
        while True:
            try:
                number_input = input("Enter the occupancy file number to analyze (e.g., 10000 for final, -1 for initial): ").strip()
                number = int(number_input)
                break
            except ValueError:
                print("Please enter a valid integer.")
        
        print(f"Analyzing occupancy files with number: {number}")
        create_parameter_sweep_visualization(runs_dir, number=number)
        print("Visualization complete!")
        
    elif mode_choice == '2':
        # Create grids for all time steps
        create_parameter_sweep_visualization(runs_dir, process_all_times=True)
        print("All visualizations complete!")
        
    elif mode_choice == '3':
        # View START configuration heatmaps
        while True:
            try:
                time_input = input("Enter time step for START configs (-1 for initial, number for final step): ").strip()
                time_step = int(time_input)
                break
            except ValueError:
                print("Please enter a valid integer.")
        
        save_choice = input("Save images to file? (y/n): ").strip().lower()
        save_dir = "analysis/start_heatmaps" if save_choice == 'y' else None
        
        print_multiple_heatmaps(runs_dir, time_step=time_step, save_dir=save_dir, prefix_filter="START_")
        print("START heatmaps complete!")
        
    elif mode_choice == '4':
        # View all heatmaps in directory
        while True:
            try:
                time_input = input("Enter time step to visualize (-1 for initial, number for specific step): ").strip()
                time_step = int(time_input)
                break
            except ValueError:
                print("Please enter a valid integer.")
        
        save_choice = input("Save images to file? (y/n): ").strip().lower()
        save_dir = "analysis/all_heatmaps" if save_choice == 'y' else None
        
        print_multiple_heatmaps(runs_dir, time_step=time_step, save_dir=save_dir)
        print("Directory heatmaps complete!")
        
    elif mode_choice == '5':
        # View single heatmap from file path
        file_path = input("Enter path to occupancy .dat file: ").strip()
        if os.path.exists(file_path):
            save_choice = input("Save image to file? (y/n): ").strip().lower()
            save_path = None
            if save_choice == 'y':
                save_path = input("Enter save path (or press Enter for auto-name): ").strip()
                if not save_path:
                    filename = os.path.basename(file_path).replace('.dat', '.png')
                    save_path = f"analysis/{filename}"
            
            print_single_heatmap(file_path=file_path, save_path=save_path)
            print("Single heatmap complete!")
        else:
            print(f"File not found: {file_path}")
    
    elif mode_choice == '6':
        # View time evolution of single configuration
        dir_path = input("Enter path to configuration directory: ").strip()
        if os.path.exists(dir_path):
            save_choice = input("Save images and animation? (y/n): ").strip().lower()
            save_dir = "analysis/time_evolution" if save_choice == 'y' else None
            
            show_choice = input("Show individual frames? (y/n): ").strip().lower()
            show_individual = show_choice == 'y'
            
            visualize_time_evolution(dir_path, save_dir=save_dir, show_individual=show_individual)
            print("Time evolution visualization complete!")
        else:
            print(f"Directory not found: {dir_path}")
