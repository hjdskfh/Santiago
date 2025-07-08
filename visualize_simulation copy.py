import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re

def extract_parameters_from_folder(folder_name):
    """
    Extract density and tumbling rate from folder name
    Assumes folder naming convention like: run_d0.5_t0.1 or similar
    Modify this function based on your naming convention
    """
    # Example patterns to match:
    # density_0.5_tumble_0.1
    # d0.5_t0.1_time10000
    # You can modify this regex based on your naming convention
    
    density_match = re.search(r'd([0-9]*\.?[0-9]+)', folder_name)
    tumble_match = re.search(r't([0-9]*\.?[0-9]+)', folder_name)
    time_match = re.search(r'time([0-9]*\.?[0-9]+)', folder_name)
    
    if density_match and tumble_match:
        density = float(density_match.group(1))
        tumble_rate = float(tumble_match.group(1))
        total_time = float(time_match.group(1)) if time_match else None
        
        return density, tumble_rate, total_time
    
    return None, None, None

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

def create_parameter_sweep_visualization(runs_dir='runs'):
    """
    Create comprehensive visualizations for parameter sweep results
    """
    # Create analysis directory if it doesn't exist
    if not os.path.exists('analysis'):
        os.makedirs('analysis')
        print("Created 'analysis' directory for output files")
    
    # Look for simulation results in the specified runs directory
    if not os.path.exists(runs_dir):
        print(f"No '{runs_dir}' directory found!")
        return
    
    # Extract date from runs_dir if it contains timestamp
    run_date = ""
    run_date_clean = ""
    if "run_" in runs_dir:
        # Extract the timestamp part from path like "runs/run_20250702_143025"
        timestamp_part = os.path.basename(runs_dir)
        if timestamp_part.startswith("run_"):
            date_str = timestamp_part[4:]  # Remove "run_" prefix
            try:
                # Convert YYYYMMDD_HHMMSS to readable format
                from datetime import datetime
                dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                run_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                run_date_clean = dt.strftime("%Y%m%d_%H%M%S")
            except:
                run_date = date_str  # Fallback to raw string
                run_date_clean = date_str.replace(":", "").replace(" ", "_")
    
    print(f"Processing run from: {run_date if run_date else 'unknown date'}")
    
    # Find all folders within runs directory that contain simulation results
    folders = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) 
               if os.path.isdir(os.path.join(runs_dir, f)) and 
               any(f.endswith('.dat') for f in os.listdir(os.path.join(runs_dir, f)) if f.startswith('Occupancy'))]
    
    if not folders:
        print(f"No simulation folders found in '{runs_dir}'!")
        return
    
    # Collect data from all runs
    results = []
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        # Extract parameters from folder name (remove the runs/ prefix)
        folder_name = os.path.basename(folder)
        density, tumble_rate, total_time = extract_parameters_from_folder(folder_name)
        
        if density is None or tumble_rate is None:
            print(f"Could not extract parameters from {folder}, skipping...")
            continue
        
        # Find the final occupancy file
        occupancy_files = glob.glob(f"{folder}/Occupancy_*.dat")
        if not occupancy_files:
            continue
        
        # Use the last timestep file
        final_file = max(occupancy_files)
        
        try:
            occupancy_data = np.loadtxt(final_file)
            metrics = calculate_metrics(occupancy_data)
            
            result = {
                'folder': folder,
                'density': density,
                'tumble_rate': tumble_rate,
                'totaltime': total_time,
                'occupancy_data': occupancy_data,
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue
    
    if not results:
        print("No valid results found!")
        return
    
    # Create visualizations
    create_individual_heatmaps(results)
    create_comparison_grid(results, run_date, run_date_clean)

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
        
        # Create individual heatmap
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        im = ax.imshow(data.T, cmap='viridis', origin='lower', aspect='equal')
        ax.set_title(f'Density={density:.2f}, Tumbling Rate={tumble_rate:.3f}, Time={totaltime}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        plt.colorbar(im, ax=ax, label='Occupancy Level')
        
        # Save individual heatmap in the run folder
        folder_name = os.path.basename(folder)
        output_file = f'{folder}/heatmap_{folder_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory
        
    print(f"Created {len(results)} individual heatmaps")

def create_comparison_grid(results, run_date="", run_date_clean=""):
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
    
    # Find global min/max for consistent color scaling
    all_data = [r['occupancy_data'] for r in results]
    global_min = min(np.min(data) for data in all_data)
    global_max = max(np.max(data) for data in all_data)
    
    # Create a dictionary for quick lookup of results by (density, tumble_rate)
    result_dict = {(r['density'], r['tumble_rate']): r for r in results}
    
    # Fill the grid organized by density (rows) and tumble rate (columns)
    for row, density in enumerate(densities):
        for col, tumble_rate in enumerate(tumble_rates):
            if (density, tumble_rate) in result_dict:
                result = result_dict[(density, tumble_rate)]
                data = result['occupancy_data']
                
                im = axes[row, col].imshow(data.T, cmap='viridis', origin='lower', 
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
        axes[row, 0].set_ylabel(rf"Density $\rho$ = {density:.2f}", fontsize=12, fontweight='bold')
    
    for col, tumble_rate in enumerate(tumble_rates):
        axes[0, col].set_xlabel(rf"Tumble Rate $\alpha$ = {tumble_rate:.3f}", fontsize=12, fontweight='bold')
        axes[0, col].xaxis.set_label_position('top')
    
    # Add a common colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Occupancy Level')
    cbar.set_label('Occupancy Level', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Create title with date information
    # Get the total time from the first result (assuming all have the same total time)
    first_result = next(iter(results), None)
    total_time = first_result['totaltime'] if first_result and first_result['totaltime'] is not None else "Unknown"
    
    # Add title as text on the figure for better control
    plt.figtext(0.5, 0.95, 'Parameter Sweep Comparison Grid', 
                fontsize=40, fontweight='bold', ha='center')
    if run_date:
        plt.figtext(0.5, 0.92, f'Time: {total_time} steps, Run Date: {run_date}', 
                fontsize=18, ha='center')
    else:
        plt.figtext(0.5, 0.89, f'Run Date: {run_date}', 
                    fontsize=18, ha='center', style='italic')
    
    # Create filename with date information
    if run_date_clean:
        filename = f'analysis/comparison_grid_{run_date_clean}.png'
    elif run_date:
        # Clean the date string for filename (replace spaces and colons)
        clean_date = run_date.replace(' ', '_').replace(':', '')
        filename = f'analysis/comparison_grid_{clean_date}.png'
    else:
        filename = 'analysis/comparison_grid.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison grid saved to '{filename}'")
    # plt.show()  # Comment out to only save, not display

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
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
    
    print("Starting parameter sweep visualization...")
    create_parameter_sweep_visualization(runs_dir)
    print("Visualization complete!")
