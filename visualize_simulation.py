import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime

from postprocessing.helper import create_discrete_colormap, extract_parameters_from_folder, calculate_metrics, load_occupancy_data, process_folder_for_sweep, find_files_in_directory

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
    
    # Create filename with date information
    if run_date:
        filename = f'analysis/comp_{run_date}__{number if number is not None else "unknown"}.png'
    else:
        filename = f'analysis/comp_{number if number is not None else "unknown"}.png'
    
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to '{filename}'")
    # plt.show()  # Comment out to only save, not display

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

def visualize_time_evolution(runs_dir, save_dir=None, show_individual=False):
    """
    Visualize the time evolution of a single configuration directory
    
    Args:
        runs_dir: Directory containing occupancy files (e.g., runs/your_run_directory/)
        save_dir: Optional directory to save images and animation
        show_individual: Whether to display each time step individually
    """
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
            # Handle both flattened (1D) and proper 2D format
            if data.ndim == 1 and len(data) == 4000:  # 100 * 40 = 4000
                data = data.reshape(40, 100)
            all_data.append(data)
            
            if show_individual:
                title = f"Time Evolution: {dir_name}\nStep {time_step}"
                if density is not None and tumble_rate is not None:
                    title = f"Density={density:.2f}, Tumble Rate={tumble_rate:.3f}\nTime Step: {time_step}"
                
                print_single_heatmap(data=data, title=title, show=show_individual)
                
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

def print_moving_particles(runs_dir="runs"):
    """
    Analyze and print movement statistics from simulation results, with plots
    """
    results = find_files_in_directory(runs_dir, pattern="movement_stats.txt")
    
    if not results:
        print("No movement statistics files found!")
        return

    run_date = ""
    if "run_" in runs_dir:
        timestamp = os.path.basename(runs_dir)[4:]  # Remove "run_" prefix
        try:
            run_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S")
        except:
            run_date = timestamp.replace(":", "").replace(" ", "_")
    
    # Create analysis directory for plots
    os.makedirs('analysis', exist_ok=True)
    
    all_data = []  # Store all data for combined plots
    
    for subdir_name, full_path, files in results:
        # Extract parameters from directory name
        density, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(subdir_name)
        
        if density is None or tumble_rate is None:
            continue
            
        # Load movement statistics file
        stats_file = os.path.join(full_path, "movement_stats.txt")
        if os.path.exists(stats_file):
            try:
                # Read the movement statistics
                data = np.loadtxt(stats_file, skiprows=1)  # Skip header
                if data.size > 0:
                    if data.ndim == 1:  # Single row
                        timesteps = np.array([data[0]])
                        moving_counts = np.array([data[1]])
                    else:  # Multiple rows
                        timesteps = data[:, 0]
                        moving_counts = data[:, 1]
                    
                    # Store data for plotting
                    all_data.append({
                        'name': subdir_name,
                        'density': density,
                        'tumble_rate': tumble_rate,
                        'gamma': gamma,
                        'g': g,
                        'timesteps': timesteps,
                        'moving_counts': moving_counts
                    })
     
                    # Create individual plot for this simulation
                    create_individual_movement_plot(timesteps, moving_counts, subdir_name, 
                                                  density, tumble_rate, gamma, g, run_date)
                
            except Exception as e:
                print(f"Error reading {stats_file}: {e}")
    
    if all_data:
        # Create combined plots
        create_combined_movement_plots(all_data, run_date)
        print(f"\nMovement analysis complete! Created {len(all_data)} individual plots and combined plots.")
    else:
        print("No valid movement data found!")

def create_individual_movement_plot(timesteps, moving_counts, name, density, tumble_rate, gamma, g, run_date):
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
    
    # Save plot with optimized settings
    filename = f'analysis/movement_{run_date}_{name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')  # Reduced DPI for smaller files
    plt.close()
    print(f"Individual movement plot saved: {filename}")

def create_combined_movement_plots(all_data, run_date=""):
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
    if run_date:
        filename = f'analysis/movement_{run_date}_all_trajectories.png'
    else:
        filename = 'analysis/movement_all_trajectories.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined trajectories plot saved: {filename}")
    
    # Plot 2: Average movement vs density (if we have multiple densities)
    densities = sorted(list(set(d['density'] for d in all_data)))
    tumble_rates = sorted(list(set(d['tumble_rate'] for d in all_data)))
    
    if len(tumble_rates) > 1:
        create_movement_plots_by_tumble_rate(all_data, run_date)

def create_movement_plots_by_tumble_rate(all_data, run_date=""):
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
        if run_date:
            filename = f'analysis/movement_{run_date}_tumble_rate_{tumble_rate:.3f}.png'
        else:
            filename = f'analysis/movement_tumble_rate_{tumble_rate:.3f}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Movement plot for tumble rate {tumble_rate:.3f} saved: {filename}")

def visualize_density_stack(runs_dir, save_dir=None, show_individual=True):
    """
    Visualize density files by stacking them with the first line at the bottom
    to create a proper diagram representation.
    
    Args:
        runs_dir: Directory containing density files
        save_dir: Optional directory to save images
        show_individual: Whether to display each density diagram individually
    """
    os.makedirs('analysis', exist_ok=True)
    
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Find density files (try both uppercase and lowercase)
    density_files = glob.glob(f"{runs_dir}/**/Density_*.dat", recursive=True)
    if not density_files:
        density_files = glob.glob(f"{runs_dir}/**/density_*.dat", recursive=True)
    if not density_files:
        print(f"No density files found in {runs_dir}")
        return
    
    print(f"Found {len(density_files)} density files")
    
    for density_file in density_files:
        try:
            # Load density data
            raw_data = np.loadtxt(density_file)
            
            # Check if data is 1D or 2D and handle accordingly
            if raw_data.ndim == 1:
                # 1D density profile - plot as line
                x = np.arange(len(raw_data))
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                ax.plot(x, raw_data, 'b-', linewidth=2)
                ax.set_xlabel('Position', fontsize=12)
                ax.set_ylabel('Density Value', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Extract info from filename
                filename = os.path.basename(density_file)
                dir_name = os.path.basename(os.path.dirname(density_file))
                
                # Try to extract parameters from directory or filename
                density_param, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
                
                # Create title
                title = f"Density Profile: {filename}\n"
                if density_param is not None and tumble_rate is not None:
                    title += f"Density={density_param:.3f}, Tumble Rate={tumble_rate:.3f}"
                else:
                    title += f"Directory: {dir_name}"
                
                ax.set_title(title, fontsize=14)
                
                # Add statistics as text
                mean_val = np.mean(raw_data)
                std_val = np.std(raw_data)
                min_val = np.min(raw_data)
                max_val = np.max(raw_data)
                ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            else:
                # 2D density heatmap - original behavior
                data = raw_data
                if len(raw_data) == 4000:
                    data = raw_data.reshape(40, 100)
                
                # Stack the data with first line at bottom (flip vertically)
                stacked_data = np.flipud(data)
                
                # Create the visualization
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Use a color map suitable for density data
                im = ax.imshow(stacked_data, cmap='viridis', origin='lower', aspect='equal')
                
                # Extract info from filename
                filename = os.path.basename(density_file)
                dir_name = os.path.basename(os.path.dirname(density_file))
                
                # Try to extract parameters from directory or filename
                density_param, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
                
                # Create title
                title = f"Density Distribution: {filename}\n"
                if density_param is not None and tumble_rate is not None:
                    title += f"Density={density_param:.3f}, Tumble Rate={tumble_rate:.3f}"
                else:
                    title += f"Directory: {dir_name}"
                
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('X Position', fontsize=12)
                ax.set_ylabel('Y Position (stacked, first line at bottom)', fontsize=12)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, label='Density Value')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, linestyle='--')
            
            # Save if requested
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_filename = filename.replace('.dat', '_stacked.png')
                save_path = os.path.join(save_dir, save_filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Density stack saved to: {save_path}")
            
            if show_individual:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error processing {density_file}: {e}")
            continue
    
    print(f"Density stack visualization complete! Processed {len(density_files)} files.")

def visualize_density_time_evolution(runs_dir, save_dir=None, show_individual=False):
    """
    Visualize the time evolution of density files for each parameter combination.
    
    Args:
        runs_dir: Directory containing density files
        save_dir: Optional directory to save images
        show_individual: Whether to display each time step individually
    """
    os.makedirs('analysis', exist_ok=True)
    
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Find all density files
    density_files = glob.glob(f"{runs_dir}/**/Density_*.dat", recursive=True)
    if not density_files:
        density_files = glob.glob(f"{runs_dir}/**/density_*.dat", recursive=True)
    if not density_files:
        print(f"No density files found in {runs_dir}")
        return
    
    print(f"Found {len(density_files)} density files")
    
    # Group files by parameter combination (directory)
    param_groups = {}
    for density_file in density_files:
        dir_path = os.path.dirname(density_file)
        dir_name = os.path.basename(dir_path)
        
        if dir_name not in param_groups:
            param_groups[dir_name] = []
        
        # Extract time step from filename
        filename = os.path.basename(density_file)
        match = re.search(r'Density_(\d+)\.dat', filename)
        if match:
            time_step = int(match.group(1))
            param_groups[dir_name].append((time_step, density_file))
    
    # Process each parameter combination
    for dir_name, files in param_groups.items():
        if len(files) < 2:
            print(f"Skipping {dir_name}: only {len(files)} density file(s) found")
            continue
        
        # Sort by time step
        files.sort(key=lambda x: x[0])
        
        print(f"Processing {dir_name}: {len(files)} time steps from {files[0][0]} to {files[-1][0]}")
        
        # Extract parameters
        density_param, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
        
        # Load all data
        all_data = []
        time_steps = []
        for time_step, file_path in files:
            try:
                raw_data = np.loadtxt(file_path)
                
                # Reshape if needed
                if raw_data.ndim == 1 and len(raw_data) == 4000:
                    data = raw_data.reshape(40, 100)
                else:
                    data = raw_data
                
                # Stack with first line at bottom
                stacked_data = np.flipud(data)
                all_data.append(stacked_data)
                time_steps.append(time_step)
                
                # Show individual frames if requested
                if show_individual:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    im = ax.imshow(stacked_data, cmap='viridis', origin='lower', aspect='equal')
                    
                    title = f"Density Evolution: {dir_name}\nTime Step: {time_step}"
                    if density_param is not None and tumble_rate is not None:
                        title = f"Density={density_param:.3f}, Tumble Rate={tumble_rate:.3f}\nTime Step: {time_step}"
                    
                    ax.set_title(title, fontsize=14)
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position (stacked)')
                    plt.colorbar(im, ax=ax, label='Density Value')
                    
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"{dir_name}_density_step_{time_step}.png")
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        plt.show()
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Create evolution grid for this parameter combination
        if len(all_data) >= 4:
            create_density_evolution_grid(all_data, time_steps, dir_name, density_param, tumble_rate, save_dir)
    
    print("Density time evolution visualization complete!")

def create_density_evolution_grid(all_data, time_steps, dir_name, density, tumble_rate, save_dir=None):
    """Create a grid showing density evolution over time for one parameter combination"""
    
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
    
    # Find global min/max for consistent color scaling
    global_min = np.min([np.min(data) for data in all_data])
    global_max = np.max([np.max(data) for data in all_data])
    
    # Plot each selected time point
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        
        time_step = time_steps[idx]
        data = all_data[idx]
        
        # Check if data is 1D or 2D and plot accordingly
        if len(data.shape) == 1:
            # 1D density profile
            x = np.arange(len(data))
            axes[row, col].plot(x, data, 'b-', linewidth=1)
            axes[row, col].set_title(f"Step {time_step}", fontsize=12)
            axes[row, col].set_xlabel('Position')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_ylim(global_min, global_max)
        else:
            # 2D density heatmap
            im = axes[row, col].imshow(data, cmap='viridis', origin='lower', 
                                      aspect='equal', vmin=global_min, vmax=global_max)
            axes[row, col].set_title(f"Step {time_step}", fontsize=12)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_points, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Add colorbar only if we have 2D data
    if len(all_data[0].shape) > 1:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, label='Density Value')
    else:
        # For 1D data, just use tight layout
        plt.tight_layout()
    
    # Add title
    if density is not None and tumble_rate is not None:
        title = f"Density Evolution: ρ={density:.3f}, α={tumble_rate:.3f}"
    else:
        title = f"Density Evolution: {dir_name}"
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"density_evolution_{dir_name}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Density evolution grid saved to: {save_dir}")
    else:
        plt.show()
    
    plt.close()

def create_density_comparison_grid(runs_dir, save_dir=None):
    """
    Create a comparison grid of multiple density files, all stacked properly.
    
    Args:
        runs_dir: Directory containing density files
        save_dir: Optional directory to save the comparison grid
    """
    os.makedirs('analysis', exist_ok=True)
    
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Find density files and organize by parameters (try both uppercase and lowercase)
    density_files = glob.glob(f"{runs_dir}/**/Density_*.dat", recursive=True)
    if not density_files:
        density_files = glob.glob(f"{runs_dir}/**/density_*.dat", recursive=True)
    if not density_files:
        print(f"No density files found in {runs_dir}")
        return
    
    # Extract parameters and organize data
    file_data = []
    for density_file in density_files:
        try:
            dir_name = os.path.basename(os.path.dirname(density_file))
            density_param, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
            
            if density_param is not None and tumble_rate is not None:
                # Load and process data
                raw_data = np.loadtxt(density_file)
                if raw_data.ndim == 1 and len(raw_data) == 4000:
                    data = raw_data.reshape(40, 100)
                else:
                    data = raw_data
                
                # Stack with first line at bottom
                stacked_data = np.flipud(data)
                
                file_data.append({
                    'density': density_param,
                    'tumble_rate': tumble_rate,
                    'data': stacked_data,
                    'filename': os.path.basename(density_file),
                    'dir_name': dir_name
                })
                
        except Exception as e:
            print(f"Error processing {density_file}: {e}")
            continue
    
    if not file_data:
        print("No valid density data found!")
        return
    
    # Get unique parameter values
    densities = sorted(list(set(item['density'] for item in file_data)))
    tumble_rates = sorted(list(set(item['tumble_rate'] for item in file_data)))
    
    n_rows = len(densities)
    n_cols = len(tumble_rates)
    
    print(f"Creating density comparison grid: {n_rows} densities × {n_cols} tumble rates")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    
    # Handle different subplot arrangements
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Find global min/max for consistent color scaling
    all_data = [item['data'] for item in file_data]
    global_min = np.min([np.min(data) for data in all_data])
    global_max = np.max([np.max(data) for data in all_data])
    
    # Create lookup dictionary
    data_dict = {(item['density'], item['tumble_rate']): item for item in file_data}
    
    # Fill the grid
    for row, density in enumerate(densities):
        for col, tumble_rate in enumerate(tumble_rates):
            if (density, tumble_rate) in data_dict:
                item = data_dict[(density, tumble_rate)]
                data = item['data']
                
                im = axes[row, col].imshow(data, cmap='viridis', origin='lower', 
                                          aspect='equal', vmin=global_min, vmax=global_max)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                
                # Add mean and std as text
                mean_val = np.mean(data)
                std_val = np.std(data)
                axes[row, col].text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                                   transform=axes[row, col].transAxes, 
                                   verticalalignment='top', fontsize=8, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[row, col].text(0.5, 0.5, 'No Data', 
                                   transform=axes[row, col].transAxes, 
                                   horizontalalignment='center', verticalalignment='center',
                                   fontsize=12)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    
    # Add labels
    for row, density in enumerate(densities):
        axes[row, 0].set_ylabel(f"{density:.2f}", fontsize=14, fontweight='bold')
    
    for col, tumble_rate in enumerate(tumble_rates):
        axes[0, col].set_xlabel(f"{tumble_rate:.3f}", fontsize=14, fontweight='bold')
        axes[0, col].xaxis.set_label_position('top')
    
    # Add axis titles
    fig.text(0.02, 0.5, 'Density ρ', fontsize=20, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.5, 0.95, 'Tumble Rate α', fontsize=20, fontweight='bold', 
             ha='center', va='center')
    
    # Add colorbar
    fig.subplots_adjust(left=0.08, right=0.88, top=0.88, bottom=0.1)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Density Value')
    cbar.set_label('Density Value', fontsize=16, fontweight='bold')
    
    # Add main title
    plt.figtext(0.5, 0.98, 'Density Distribution Comparison (Stacked)', 
                fontsize=24, fontweight='bold', ha='center')
    
    # Save the comparison grid
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'density_comparison_grid.png')
    else:
        save_path = 'analysis/density_comparison_grid.png'
    
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Density comparison grid saved to: {save_path}")
    plt.close()



def visualize_density_evolution_stacked(runs_dir, save_dir=None, show_individual=True, create_comparison_grid=False, run_date=None):
    """
    Create 2D stacked visualization of 1D density evolution over time.
    Each 1D density profile becomes a horizontal line, stacked vertically by time.
    
    Args:
        runs_dir: Directory containing density files
        save_dir: Optional directory to save images
        show_individual: Whether to display each parameter combination individually
        create_comparison_grid: Whether to create a comparison grid of all parameter combinations
    """
    os.makedirs('analysis', exist_ok=True)
    
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Find all density files
    density_files = glob.glob(f"{runs_dir}/**/Density_*.dat", recursive=True)
    if not density_files:
        density_files = glob.glob(f"{runs_dir}/**/density_*.dat", recursive=True)
    if not density_files:
        print(f"No density files found in {runs_dir}")
        return
    
    print(f"Found {len(density_files)} density files")
    
    # Group files by parameter combination (directory)
    param_groups = {}
    for density_file in density_files:
        dir_path = os.path.dirname(density_file)
        dir_name = os.path.basename(dir_path)
        
        if dir_name not in param_groups:
            param_groups[dir_name] = []
        
        # Extract time step from filename
        filename = os.path.basename(density_file)
        match = re.search(r'[Dd]ensity_(\d+)\.dat', filename)
        if match:
            time_step = int(match.group(1))
            param_groups[dir_name].append((time_step, density_file))
    
    # Store processed data for comparison grid
    all_processed_data = []
    
    # Process each parameter combination
    for dir_name, files in param_groups.items():
        if len(files) < 2:
            print(f"Skipping {dir_name}: only {len(files)} density file(s) found")
            continue
        
        # Sort by time step
        files.sort(key=lambda x: x[0])
        
        print(f"Processing {dir_name}: {len(files)} time steps from {files[0][0]} to {files[-1][0]}")
        
        # Extract parameters
        density_param, tumble_rate, total_time, gamma, g = extract_parameters_from_folder(dir_name)
        
        # Load all 1D density profiles
        density_matrix = []
        time_steps = []
        
        for time_step, file_path in files:
            try:
                raw_data = np.loadtxt(file_path)
                
                # Ensure we have 1D data
                if raw_data.ndim == 1:
                    density_matrix.append(raw_data)
                    time_steps.append(time_step)
                else:
                    print(f"Skipping {file_path}: expected 1D data but got {raw_data.shape}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if len(density_matrix) < 2:
            print(f"Not enough valid 1D density files for {dir_name}")
            continue
        
        # Convert to 2D array: rows = time steps, columns = spatial positions
        density_2d = np.array(density_matrix)
        
        print(f"Created 2D density array: {density_2d.shape} (time_steps × spatial_positions)")
        
        # Store data for comparison grid
        if create_comparison_grid and density_param is not None and tumble_rate is not None:
            all_processed_data.append({
                'density': density_param,
                'tumble_rate': tumble_rate,
                'dir_name': dir_name,
                'density_2d': density_2d,
                'time_steps': time_steps,
                'gamma': gamma,
                'g': g
            })
        
        # Create individual visualization
        if show_individual or not create_comparison_grid:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            
            # Create the heatmap
            im = ax.imshow(density_2d, cmap='viridis', aspect='auto', origin='lower', 
                          interpolation='nearest')
            
            # Set up axes
            ax.set_xlabel('Spatial Position', fontsize=14)
            ax.set_ylabel('Time Step', fontsize=14)
            
            # Set y-tick labels to actual time steps (subsample if too many)
            n_ticks = min(10, len(time_steps))
            tick_indices = np.linspace(0, len(time_steps)-1, n_ticks, dtype=int)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels([str(time_steps[i]) for i in tick_indices])
            
            # Create title
            if density_param is not None and tumble_rate is not None:
                title = f"Density Evolution Over Time\n"
                title += f"ρ={density_param:.3f}, α={tumble_rate:.3f}"
                if gamma is not None:
                    title += f", γ={gamma:.3f}"
                if g is not None:
                    title += f", g={g:.3f}"
            else:
                title = f"Density Evolution: {dir_name}"
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Density Value')
            cbar.set_label('Density Value', fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics as text
            mean_val = np.mean(density_2d)
            std_val = np.std(density_2d)
            min_val = np.min(density_2d)
            max_val = np.max(density_2d)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save if requested
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                if run_date:
                    safe_dir_name = dir_name.replace('/', '_').replace('\\', '_')
                    save_path = os.path.join(save_dir, f"density_evolution_stacked_{safe_dir_name}_{run_date}.png")
                else:
                    safe_dir_name = dir_name.replace('/', '_').replace('\\', '_')
                    save_path = os.path.join(save_dir, f"density_evolution_stacked_{safe_dir_name}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved stacked density evolution: {save_path}")
            
            if show_individual:
                plt.show()
            else:
                plt.close()
    
    # Create comparison grid if requested
    if create_comparison_grid and all_processed_data:
        create_density_evolution_comparison_grid(all_processed_data, save_dir, run_date)
    print("Density evolution stacked visualization complete!")

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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
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
            if (density, tumble_rate) in data_dict:
                item = data_dict[(density, tumble_rate)]
                density_2d = item['density_2d']
                time_steps = item['time_steps']
                
                # Create the heatmap
                im = axes[row, col].imshow(density_2d, cmap='viridis', aspect='auto', 
                                          origin='lower', interpolation='nearest',
                                          vmin=global_min, vmax=global_max)
                
                # Set up y-tick labels to show time steps (subsample if too many)
                n_ticks = min(5, len(time_steps))
                tick_indices = np.linspace(0, len(time_steps)-1, n_ticks, dtype=int)
                axes[row, col].set_yticks(tick_indices)
                axes[row, col].set_yticklabels([str(time_steps[i]) for i in tick_indices], fontsize=8)
                
                # Remove x-ticks for cleaner look
                axes[row, col].set_xticks([])
                
                # Add parameter values as subtitle
                axes[row, col].set_title(f"ρ={density:.2f}, α={tumble_rate:.3f}", fontsize=10)
                
                # Add basic statistics as text
                mean_val = np.mean(density_2d)
                std_val = np.std(density_2d)
                axes[row, col].text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                                   transform=axes[row, col].transAxes, 
                                   verticalalignment='top', fontsize=8, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                # No data for this parameter combination
                axes[row, col].text(0.5, 0.5, 'No Data', 
                                   transform=axes[row, col].transAxes, 
                                   horizontalalignment='center', verticalalignment='center',
                                   fontsize=12)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
    
    # Add row and column labels
    for row, density in enumerate(densities):
        axes[row, 0].set_ylabel(f"ρ={density:.2f}", fontsize=12, fontweight='bold')
    
    for col, tumble_rate in enumerate(tumble_rates):
        axes[-1, col].set_xlabel(f"α={tumble_rate:.3f}", fontsize=12, fontweight='bold')
    
    # Add overall axis labels
    fig.text(0.02, 0.5, 'Density ρ', fontsize=18, fontweight='bold', 
             rotation=90, va='center', ha='center')
    fig.text(0.5, 0.02, 'Tumble Rate α', fontsize=18, fontweight='bold', 
             ha='center', va='center')
    
    # Add colorbar
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Density Value')
    cbar.set_label('Density Value', fontsize=14, fontweight='bold')
    
    # Add main title
    plt.figtext(0.5, 0.97, 'Density Evolution Comparison Grid (2D Stacked)', 
                fontsize=20, fontweight='bold', ha='center')
    plt.figtext(0.5, 0.94, 'Time evolution shown as stacked 1D profiles (vertical: time, horizontal: space)', 
                fontsize=14, ha='center', style='italic')
    
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
    
    # Extract run date from directory name for use in output filenames
    run_date = ""
    if "run_" in runs_dir:
        timestamp = os.path.basename(runs_dir)[4:]  # Remove "run_" prefix
        try:
            run_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y%m%d_%H%M%S")
        except:
            run_date = timestamp.replace(":", "").replace(" ", "_")
    
    print("Starting simulation visualization...")
    print(f"Run date identifier: {run_date}")
    
    # Show options to user
    print("\nChoose visualization mode:")
    print("1. Parameter sweep comparison grid (specific time step)")
    print("2. Parameter sweep grids for ALL time steps")
    print("3. View START configuration heatmaps")
    print("4. View all heatmaps in directory") 
    print("5. View single heatmap from file path")
    print("6. View time evolution of single configuration")
    print("7. Analyze movement statistics")
    print("12. Create 2D stacked density evolution (1D profiles → 2D time evolution)")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1-7, 12): ").strip()
            if mode_choice in ['1', '2', '3', '4', '5', '6', '7', '12']:
                break
            else:
                print("Please enter a number from 1-7 or 12.")
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
    
    elif mode_choice == '7':
        # Analyze movement statistics
        print("Analyzing movement statistics...")
        print_moving_particles(runs_dir)
    
    elif mode_choice == '12':
        # Create 2D stacked density evolution (restored option 12)
        save_choice = input("Save stacked density evolution images to file? (y/n): ").strip().lower()
        save_dir = "analysis/density_stacked_evolution" if save_choice == 'y' else None
        
        show_choice = input("Show individual parameter combinations? (y/n): ").strip().lower()
        show_individual = show_choice == 'y'
        
        print("Creating 2D stacked density evolution visualization...")
        visualize_density_evolution_stacked(runs_dir, save_dir=save_dir, show_individual=show_individual)
        print("2D stacked density evolution complete!")