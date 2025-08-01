import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.linalg import svd

from postprocessing.engine import create_discrete_colormap, find_files_in_directory, \
    load_occupancy_data, extract_parameters_from_folder, process_folder_for_sweep, \
    create_comparison_grid, create_density_evolution_comparison_grid, \
    create_individual_movement_plot, create_combined_movement_plots, \
    compute_density_profiles_by_step, plot_density_derivative_grid, create_time_evolution_grid, \
    nw_kernel_regression, compute_flux_profiles_by_step, check_if_at_integration_points_equal, find_move_prob_file

from postprocessing.helper import find_all_roots

def create_parameter_sweep_visualization(runs_dir='runs', number=1, process_all_times=False, save_dir=None):
    """Create comprehensive visualizations for parameter sweep results."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
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
            create_parameter_sweep_visualization(runs_dir, number=time_step, process_all_times=False, save_dir=save_dir)
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
    create_comparison_grid(results, save_dir=save_dir, run_date=run_date, number=number)

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
    occupancy_files = glob.glob(f"{runs_dir}/Occupancy_*.dat")
    if not occupancy_files:
        print(f"No occupancy files found in {runs_dir}")
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
    dir_name = os.path.basename(runs_dir)
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
        create_time_evolution_grid(all_data, time_files, dir_name, density, tumble_rate, save_dir)
    
    print(f"Time evolution visualization complete! Processed {len(all_data)} time steps.")

def print_moving_particles(runs_dir="runs", save_dir=None):
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
                                                  density, tumble_rate, gamma, g, run_date, save_dir=save_dir)
                
            except Exception as e:
                print(f"Error reading {stats_file}: {e}")
    
    if all_data:
        # Create combined plots
        create_combined_movement_plots(all_data, run_date, save_dir=save_dir)
        print(f"\nMovement analysis complete! Created {len(all_data)} individual plots and combined plots.")
    else:
        print("No valid movement data found!")

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

def analyze_density_derivatives_grid(runs_dir, steps_to_include=None, smooth=True, save_choice=False, save_dir=None, method=None):
    profiles_by_step = compute_density_profiles_by_step(runs_dir, steps_to_include, smooth=smooth, method=method)
    if not profiles_by_step:
        print("No profiles found for the selected steps.")
        return
    plot_density_derivative_grid(profiles_by_step, save_choice=save_choice, save_dir=save_dir, title_prefix="Smoothed Profiles & Derivatives", method=method)

# ---- CASE: LAMBDA AND GAMMA CONSTANTS -----
def compute_gamma_lambda_constant(runs_dir, save_dir, method='diff', start_averaging_step=0, x_min=0, x_max=200):
    """ Compute gamma and lambda constants for given experimental data."""
   
    gam_exp = []
    lam_exp = []
    erf_exp = []
    cov_exp = []

    rho_min = 1
    rho_max = 1.75
    rho_est_arr = np.linspace(rho_min, rho_max, 10)

    print("sim", "|" , "gamma", "|" , "lambda")
    # Prepare to write gamma and lambda to a file
    output_filename = os.path.join(save_dir, 'gamma_lambda_results.txt')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'w')
    output_file.write("sim\tfolder\tgamma\tlambda\n")

    # Use both start and end step for averaging
    steps_to_include = start_averaging_step
    profiles_by_step_density = compute_density_profiles_by_step(runs_dir, steps_to_include, smooth=True, method=method)
    profiles_by_step_flux = compute_flux_profiles_by_step(runs_dir, steps_to_include, smooth=True, method=method)

    for idx, (key_density, value_density) in enumerate(profiles_by_step_density.items()):
        # load data for this simulation run
        # key is (folder_name, step)
        print("key_density:", key_density)
        rho_exp, d_rho_exp, d2_rho_exp = value_density
        print(f"rho_exp: {rho_exp.shape}")
        print("Any zeros in rho_exp?", np.any(rho_exp == 0))
        plt.figure()
        plt.plot(rho_exp, label=f"rho_exp")
        plt.savefig(f"{save_dir}/rho_exp_{idx}.png")
        plt.close()

        # Get J for the same key
        J_exp = profiles_by_step_flux.get(key_density)
        if J_exp is None:
            print(f"[Warning] No flux profile for {key_density}, skipping.")
            continue

        # return contents of first MoveProb file it finds as np array
        gradU = find_move_prob_file(runs_dir)

        A = []
        B = []
        erf = []

        for rho_est in rho_est_arr:
            x_grid = np.linspace(0, 200, len(rho_exp))
            rho_moved = rho_exp - rho_est
            rho_moved_interp = interp1d(x_grid, rho_moved, kind='cubic')
            J_div_rho = J_exp / rho_exp
            J_div_rho_minus_grad_U = J_div_rho - gradU
            print("Any NaN in J_div_rho?", np.isnan(J_div_rho).any())

            # determination of integration limits
            roots = find_all_roots(rho_moved_interp, x_min, x_max, steps=1000)
            print(f"Roots found: {roots}")
            

            a = roots[0] if len(roots) > 0 else x_min
            b = roots[1] if len(roots) > 0 else x_max

            # Suppose x_grid is the grid of x values (e.g., np.linspace(0, 200, len(d_rho_exp)))
            d_rho_exp_func = interp1d(x_grid, d_rho_exp, kind='cubic', fill_value="extrapolate")
            d2_rho_exp_func = interp1d(x_grid, d2_rho_exp, kind='cubic', fill_value="extrapolate")

            check_if_at_integration_points_equal(save_dir, d_rho_exp_func, d_rho_exp, a, b)
            check_if_at_integration_points_equal(save_dir, d2_rho_exp_func, d2_rho_exp, a, b)

            #plt.plot(rho_moved_interp(np.linspace(a, b, 100)), label=f"rho_est={rho_est:.2f}")
            #plt.legend()
            #plt.show()

            # integrals for current
            J_div_rho_minus_grad_U_func = interp1d(x_grid, J_div_rho_minus_grad_U, kind='cubic', fill_value="extrapolate")
            integral_j, err = quad(J_div_rho_minus_grad_U_func, a, b)

            # values of A and B
            a0 = -(d2_rho_exp_func(b) - d2_rho_exp_func(a))
            a1 = (d_rho_exp_func(b) ** 2 - d_rho_exp_func(a) ** 2)
            # subtract the one particle current to eliminate potential influence
            b = - integral_j
            A.append([a0, a1])
            B.append(b)
        
        A = np.array(A)
        B = np.array(B)

        print("A:", A)
        print("B:", B)
        print("Any NaN in A?", np.isnan(A).any())
        print("Any NaN in B?", np.isnan(B).any())

        U, s, Vh = svd(A, full_matrices=False)

        print("U:", U)
        print("s:", s)
        print("Vh:", Vh)

        gam, lam = Vh.T @ np.diag(1 / s) @ U.T @ B
        gam_exp.append(round(gam, 2))
        lam_exp.append(round(lam, 2))


        cov = Vh @ np.diag(1 / s**2) @ Vh.T
        cov_exp.append(cov)
        erf = B - A@[gam, lam]
        erf_exp.append(erf)

        print(idx, "|" , gam, "|" , lam, 2)
        # Write to file as well, including folder name (from key_density)
        folder_name = key_density[0] if isinstance(key_density, tuple) and len(key_density) > 0 else str(key_density)
        output_file.write(f"{idx}\t{folder_name}\t{gam}\t{lam}\n")

    output_file.close()
    print(f"Gamma and lambda results written to {output_filename}")


def compute_gamma_lambda_density_dep(runs_dir, save_dir, method='diff', start_averaging_step=0, end_averaging_step=None, x_min=0, x_max=200):
    """ Compute gamma and lambda constants for given experimental data, allowing user to choose end averaging step."""
    
    a0_exp = []
    a1_exp = []

    # I added that
    rho_min = 1
    rho_max = 1.75
    rho_est_arr = np.linspace(rho_min, rho_max, 10)

    print("sim", "|" , "gamma", "|" , "lambda")
    # Prepare to write gamma and lambda to a file
    output_filename = os.path.join(save_dir, 'gamma_lambda_results.txt')
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    output_file = open(output_filename, 'w')
    output_file.write("sim\tfolder\tgamma\tlambda\n")

    # Use both start and end step for averaging
    steps_to_include = start_averaging_step
    profiles_by_step_density = compute_density_profiles_by_step(runs_dir, steps_to_include, smooth=True, method=method)
    profiles_by_step_flux = compute_flux_profiles_by_step(runs_dir, steps_to_include, smooth=True, method=method)

    for idx, (key_density, value_density) in enumerate(profiles_by_step_density.items()):
        # load data for this simulation run
        print("key_density:", key_density)
        rho_exp, d_rho_exp, d2_rho_exp = value_density
        plt.figure()
        plt.plot(rho_exp, label=f"rho_exp")
        plt.savefig(f"{save_dir}/rho_exp_{idx}.png")
        plt.close()

        J_exp = profiles_by_step_flux.get(key_density)
        if J_exp is None:
            print(f"[Warning] No flux profile for {key_density}, skipping.")
            continue

        # return contents of first MoveProb file it finds as np array
        gradU = find_move_prob_file(runs_dir)
        # alt
        rho = rho_exp[i]
        J = J_exp[i]
        gs = gs_exp[i]
        
        # non-parametric functions
        rho_non_par = lambda X: m(X, x, rho)
        j_non_par = lambda X: m(X, x, J)
        rho_g = lambda X: rho_non_par(X) * g_x(X, gs)
        
        # derivative rho
        d_rho = np.gradient(rho_non_par(x_smooth), x_smooth)
        d_rho_non_par = lambda X: m(X, x_smooth, d_rho)

        d2_rho = np.gradient(np.gradient(rho_non_par(x_smooth), x_smooth), x_smooth)
        d2_rho_non_par = lambda X: m(X, x_smooth, d2_rho)

        # plt.figure()
        # plt.plot(x, rho_non_par(x))
        # plt.plot(x, d_rho_non_par(x))
        # plt.plot(x, d2_rho_non_par(x))

        for rho_est in rho_est_arr:
            
            rho_zero = lambda X: rho_non_par(X) - rho_est
            rho_j = lambda X: j_non_par(X) / rho_non_par(X)

            # determination of integration limits
            a = brentq(rho_zero, 0, 180) # left limit
            b = brentq(rho_zero, 180, 400) # right limit

            # integrals for current
            integral_j, err = quad(rho_j, a, b)
            integral_rho_g, err = quad(g_x, a, b, args=gs)

            # values of A and B
            a0 = -(d2_rho_non_par(b) - d2_rho_non_par(a))
            a1 = (d_rho_non_par(b) ** 2 - d_rho_non_par(a) ** 2)
            b = -integral_j - integral_rho_g

            a0_exp.append(a0[0] / b)
            a1_exp.append(a1[0] / b)
        

    # System of equations

    # Matrix build
    N = len(rho_est_arr) # number of rho evaluated
    M = N_sims # number of simulations

    A = np.zeros((M*N, 2*N))

    print(len(a0_exp))

    for i in range(N): # Recorre rho
        for k in range(M): # Recorre experimentos
            A[k + i*M, 2*i] = a0_exp[k * N + i]
            A[k + i*M, 2*i+1] = a1_exp[k * N + i]

    d = np.ones(M*N)

    # Descomposición SVD: A = U @ S @ Vt

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)

    #Encontrar lambda y gamma
    # Resolver sistema A * b = d para b

    V = np.transpose(Vt)
    Ut = np.transpose(U)
    S_1 = np.diag(1/s)

    b = V @ S_1 @ Ut @ d

    eta = np.zeros(N)
    gam = np.zeros(N)

    for i in range(N):
        gam[i] = b[2 * i]
        eta[i] = b[2 * i + 1]

    cov = V @ np.diag(1 / s ** 2) @ V.T