import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime 
import bisect

from postprocessing.manager import create_parameter_sweep_visualization, \
    print_multiple_heatmaps, print_single_heatmap, visualize_time_evolution, \
    print_moving_particles, visualize_density_evolution_stacked, analyze_density_derivatives_grid, compute_gamma_lambda_constant, compute_gamma_lambda_density_dep
from postprocessing.helper import plot_file

if __name__ == "__main__":
    import time
    import sys
    
    start_time = time.time()

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

    # Create a new analysis folder for this run
    analysis_dir = f"analysis/{run_date}" if run_date else "analysis/default"
    os.makedirs(analysis_dir, exist_ok=True)

    print("Starting simulation visualization...")
    print(f"Run date identifier: {run_date}")
    print(f"Analysis output will be saved in: {analysis_dir}")

    # Show options to user
    print("\nChoose visualization mode:")
    print("1. Visualize density over parameter sweep with comparison grid (specific time step)")
    print("2. Visualize density over parameter sweep grids for ALL time steps in the given runs directory")
    print("3. View START configuration of density heatmaps")
    print("4. View all density heatmaps in directory") 
    print("5. View single density heatmap from file path")
    print("6. View time evolution of density of single configuration")
    print("7. Visualize the observable number of moving particles to determine timestep to start averaging")
    print("8. Create 2D stacked density evolution over the timesteps")
    print("9. Visualize density derivatives for smoothed densities starting from one or several averaging timesteps in a grid")
    print("10. Lambda and Gamma constant and or density dependent: Calculate lambda and gamma for smoothed densities starting from one or several averaging timesteps in a grid (smoothing activated, results printed)")

    while True:
        try:
            mode_choice = input("\nEnter your choice (1-10, or comma-separated for multiple): ").strip()
            mode_choices = [c.strip() for c in mode_choice.split(',') if c.strip() in [str(i) for i in range(1, 11)]]
            if mode_choices:
                break
            else:
                print("Please enter a number from 1-10, or a comma-separated list like 1,7,9,10.")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)
    
    for mode_choice in mode_choices:
        if mode_choice == '1':
            save_dir = os.path.join(analysis_dir, f"comparison_grids")
            while True:
                try:
                    number_input = input("Enter the occupancy file number(s) to analyze (e.g., 10000 for final, -1 for initial, or comma-separated list like 1,10,100): ").strip()
                    number_list = [int(n.strip()) for n in number_input.split(',') if n.strip()]
                    break
                except ValueError:
                    print("Please enter valid integer(s), separated by commas if multiple.")
            print(f"Analyzing occupancy files with number(s): {number_list}")
            for number in number_list:
                print(f"Creating comparison grid for step: {number}")
                create_parameter_sweep_visualization(runs_dir, number=number, save_dir=save_dir)
                print("Visualization(s) complete!")

        elif mode_choice == '2':
            create_parameter_sweep_visualization(runs_dir, process_all_times=True, save_dir=save_dir)
            print("All visualizations complete!")

        elif mode_choice == '3':
            while True:
                try:
                    time_input = input("Enter time step for START configs (-1 for initial, number for final step): ").strip()
                    time_step = int(time_input)
                    break
                except ValueError:
                    print("Please enter a valid integer.")
            save_choice = input("Save images to file? (y/n): ").strip().lower()
            save_dir = os.path.join(analysis_dir, "start_heatmaps") if save_choice == 'y' else None
            print_multiple_heatmaps(runs_dir, time_step=time_step, save_dir=save_dir, prefix_filter="START_")
            print("START heatmaps complete!")

        elif mode_choice == '4':
            while True:
                try:
                    time_input = input("Enter time step to visualize (-1 for initial, number for specific step): ").strip()
                    time_step = int(time_input)
                    break
                except ValueError:
                    print("Please enter a valid integer.")
            save_choice = input("Save images to file? (y/n): ").strip().lower()
            save_dir = os.path.join(analysis_dir, "all_heatmaps") if save_choice == 'y' else None
            print_multiple_heatmaps(runs_dir, time_step=time_step, save_dir=save_dir)
            print("Directory heatmaps complete!")
        elif mode_choice == '5':
            file_path = input("Enter path to occupancy .dat file: ").strip()
            if os.path.exists(file_path):
                save_choice = input("Save image to file? (y/n): ").strip().lower()
                save_path = None
                if save_choice == 'y':
                    save_path = input("Enter save path (or press Enter for auto-name): ").strip()
                    if not save_path:
                        filename = os.path.basename(file_path).replace('.dat', '.png')
                        save_path = os.path.join(analysis_dir, filename)
                print_single_heatmap(file_path=file_path, save_path=save_path)
                print("Single heatmap complete!")
            else:
                print(f"File not found: {file_path}")
        elif mode_choice == '6':
            use_defaults = True
            if use_defaults:
                save_choice = 'y'
                show_choice = 'n'
            else:
                save_choice = input("Save images and animation? (y/n): ").strip().lower()
                show_choice = input("Show individual frames? (y/n): ").strip().lower()

            save_dir = os.path.join(analysis_dir, "time_evolution") if save_choice == 'y' else None
            show_individual = show_choice == 'y'
            visualize_time_evolution(runs_dir, save_dir=save_dir, show_individual=show_individual)
            print("Time evolution visualization complete!")

        elif mode_choice == '7':
            print("Analyzing movement statistics...")
            analysis_dir_movement = os.path.join(analysis_dir, "moving_particles_statistics")
            print_moving_particles(runs_dir, save_dir=analysis_dir_movement)

        elif mode_choice == '8':
            use_defaults = True  # Change to False if you want to re-enable input
            if use_defaults:
                save_choice = 'y'
                show_choice = 'n'
            else:
                save_choice = input("Save stacked density evolution images to file? (y/n): ").strip().lower()
                show_choice = input("Show individual parameter combinations? (y/n): ").strip().lower()
                
            save_dir = os.path.join(analysis_dir, "density_stacked_evolution") if save_choice == 'y' else analysis_dir
            show_individual = show_choice == 'y'

            print("Creating 2D stacked density evolution visualization...")
            visualize_density_evolution_stacked(
                runs_dir,
                save_dir=save_dir,
                show_individual=show_individual,
                create_comparison_grid=True,
                run_date=run_date
            )
            print("2D stacked density evolution complete!")

        elif mode_choice == '9':
            use_defaults = True  # Change to False if you want to re-enable input
            if use_defaults:
                step_list = [5000]  # Example default steps
                kind_of_derivative = 'both'
                smooth_choice = 'y'
                kernel_width = 2  # Default kernel width factor
            else:
                while True:
                    try:
                        step_input = input("Enter the timesteps to analyze (e.g., 1000, 3000, 5000 or 1000 to 5000 in steps of 1000): ").strip()
                        if 'to' in step_input and 'step' in step_input:
                            import re
                            match = re.match(r"(\d+)\s*to\s*(\d+)\s*in steps of\s*(\d+)", step_input)
                            if match:
                                start = int(match.group(1))
                                end = int(match.group(2))
                                step_size = int(match.group(3))
                                step_list = list(range(start, end + 1, step_size))
                            else:
                                print("Invalid range format. Please use e.g. 1000 to 5000 in steps of 50.")
                                continue
                        elif 'to' in step_input:
                            match = re.match(r"(\d+)\s*to\s*(\d+)", step_input)
                            if match:
                                start = int(match.group(1))
                                end = int(match.group(2))
                                step_list = list(range(start, end + 1, 1))
                            else:
                                print("Invalid range format. Please use e.g. 1000 to 5000.")
                                continue
                        else:
                            step_list = [int(s.strip()) for s in step_input.split(",") if s.strip()]
                        if step_list:
                            break
                        else:
                            print("Please enter at least one timestep.")
                    except ValueError:
                        print("Invalid input. Please enter integers separated by commas or a range like 1000 to 5000 in steps of 50.")
                kind_of_derivative = input("Choose method for density derivative calculation (Gaussian kernel (input: kernel), finite_difference (input: diff)) or both (input: both): ").strip().lower()
                print(f"Calculating density derivatives using method: {kind_of_derivative}")
                smooth_choice = input("Do you want to smooth the density profiles? (y/n): ").strip().lower()
                while True:
                    kernel_width = input("Enter kernel width for Gaussian smoothing in factor of mu (default is 1 -> sigma, 2: 2*sigma): ").strip()
                    try:
                        kernel_width = float(kernel_width)
                        break
                    except ValueError:
                        print("Please enter a single numeric value (e.g., 1 or 2).")

            save_choice = 'y'
            save_dir = os.path.join(analysis_dir, "density_stacked_evolution") if save_choice == 'y' else None
            smooth_density = smooth_choice == 'y'

            title_steps = "steps_" + "_".join(map(str, step_list))
            if save_choice == 'y' and smooth_choice == 'y':
                save_dir = os.path.join(analysis_dir, f"grid_density_average_derivatives_mu_{kernel_width}sigma_{kind_of_derivative}_{title_steps}_smoothing")
            if save_choice == 'y' and smooth_choice == 'n':
                save_dir = os.path.join(analysis_dir, f"grid_density_average_derivatives_mu_{kernel_width}sigma_{kind_of_derivative}_{title_steps}")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            if kind_of_derivative == 'both':
                print("Calculating both Gaussian kernel and finite difference derivatives.")
                analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, mu=kernel_width, save_choice=save_choice, save_dir=save_dir, method='kernel')
                analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, mu=kernel_width, save_choice=save_choice, save_dir=save_dir, method='diff')
            else:
                print(f"Calculating {kind_of_derivative} density derivatives.")
                analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, mu=kernel_width, save_choice=save_choice, save_dir=save_dir, method=kind_of_derivative)

        elif mode_choice == '10':
        
            use_defaults = True  # Change to False if you want to re-enable input
            if use_defaults:
                start_averaging_step = 5000  # Example default steps
                density_files = glob.glob(os.path.join(runs_dir, '**', 'Density_*.dat'), recursive=True)
                if density_files:
                    # Limit to first 10 files for debug/performance
                    limited_density_files = density_files[:10]
                    steps_found = [int(re.search(r'Density_(\d+)\.dat', os.path.basename(f)).group(1)) for f in limited_density_files if re.search(r'Density_(\d+)\.dat', os.path.basename(f))]
                else:
                    raise FileNotFoundError("No density files found in the specified runs directory.")
                method_input = 'diff'  # do both methods
                lambda_choice = 'densitydep'  # default to density dependent
                set_rho_min = 1.4  # default min density
                set_rho_max = 2.7  # default max density
                set_mu = 1  # default kernel width factor
                nr_of_slices = 20  # default number of slices
            else:
                while True:
                    try:
                        step_input = input("Enter start timestep to analyze (e.g., 10000): ").strip()
                        start_averaging_step = int(step_input)
                        if start_averaging_step % 1000 == 0:
                            break
                        else:
                            print("Invalid input. Please enter a step that is a multiple of 1000.")
                    except ValueError:
                        print("Invalid input. Please enter a valid timestep.")
                method_input = input("Choose method for lambda and gamma calculation (Gaussian kernel (input: kernel), finite_difference (input: diff)) or both (input: both): ").strip().lower()
                lambda_choice = input("Choose method for lambda calculation constant (input: constant), density dependent (input: densitydep), or both (input: both): ").strip().lower()
                set_rho_min = input("Enter min density for analysis (default is 1.2): ").strip()
                set_rho_max = input("Enter max density for analysis (default is 2.5): ").strip()      
                set_mu = input("Enter kernel width for Gaussian smoothing in factor of mu (default is 1 -> sigma, 2: 2*sigma): ").strip()
                nr_of_slices = input("Enter number of slices for rho estimation (default is 20): ").strip()

            # Check if Density_avg file in first folder of runs_dir and override start_averaging_step if it exists
            folders = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, f))]
            densavg_path = os.path.join(folders[0], "Density_avg_start.dat")
            if os.path.exists(densavg_path):
                steps = sorted([
                    int(os.path.basename(f).split("_")[1].split(".")[0])
                    for f in density_files
                ])
                start_averaging_step = steps[0]  # Override to start from the beginning if Density_avg exists
                density_avg_exists = True
                print("Density_avg_start.dat found!")
            else:
                density_avg_exists = False
                print("No Density_avg_start.dat found.")

            if lambda_choice not in ['constant', 'densitydep', 'both']:
                raise ValueError(f"Invalid lambda choice: {lambda_choice}")
            if method_input not in ['kernel', 'diff', 'both']:
                raise ValueError(f"Invalid method choice: {method_input}")

            # Convert set_rho_min and set_rho_max to float if needed
            try:
                set_rho_min = float(set_rho_min)
            except Exception:
                set_rho_min = 1.2
            try:
                set_rho_max = float(set_rho_max)
            except Exception:
                set_rho_max = 2.5


            # Prepare output directories (add nr_of_slices to output and function calls)
            save_dir_diff_const = os.path.join(analysis_dir, f"results_gamma_lambda_diff_constant_rho{set_rho_min}_{set_rho_max}_start{start_averaging_step}_mu{set_mu}sigma_slices{nr_of_slices}")
            save_dir_diff_densitydep = os.path.join(analysis_dir, f"results_gamma_lambda_diff_densitydep_rho{set_rho_min}_{set_rho_max}_start{start_averaging_step}_mu{set_mu}sigma_slices{nr_of_slices}")
            save_dir_kernel_const = os.path.join(analysis_dir, f"results_gamma_lambda_kernel_constant_rho{set_rho_min}_{set_rho_max}_start{start_averaging_step}_mu{set_mu}sigma_slices{nr_of_slices}")
            save_dir_kernel_densitydep = os.path.join(analysis_dir, f"results_gamma_lambda_kernel_densitydep_rho{set_rho_min}_{set_rho_max}_start{start_averaging_step}_mu{set_mu}sigma_slices{nr_of_slices}")
            output_dirs = [
                save_dir_kernel_const, save_dir_diff_const,
                save_dir_kernel_densitydep, save_dir_diff_densitydep
            ]
            for d in output_dirs:
                os.makedirs(os.path.dirname(d), exist_ok=True)

            # Define compute and plot tasks
            compute_tasks = []
            plot_tasks = []

            if method_input in ['kernel', 'both']:
                if lambda_choice in ['constant', 'both']:
                    compute_tasks.append((compute_gamma_lambda_constant, runs_dir, save_dir_kernel_const, 'kernel', True, nr_of_slices))
                    plot_tasks.append((plot_file, runs_dir, save_dir_kernel_const))
                if lambda_choice in ['densitydep', 'both']:
                    compute_tasks.append((compute_gamma_lambda_density_dep, runs_dir, save_dir_kernel_densitydep, 'kernel', True, nr_of_slices))
                    plot_tasks.append((plot_file, runs_dir, save_dir_kernel_densitydep))

            if method_input in ['diff', 'both']:
                if lambda_choice in ['constant', 'both']:
                    compute_tasks.append((compute_gamma_lambda_constant, runs_dir, save_dir_diff_const, 'diff', False, nr_of_slices))
                    plot_tasks.append((plot_file, runs_dir, save_dir_diff_const))
                if lambda_choice in ['densitydep', 'both']:
                    compute_tasks.append((compute_gamma_lambda_density_dep, runs_dir, save_dir_diff_densitydep, 'diff', False, nr_of_slices))
                    plot_tasks.append((plot_file, runs_dir, save_dir_diff_densitydep))

            # Run compute tasks
            for func, rdir, sdir, method, use_mu, slices in compute_tasks:
                if use_mu:
                    func(rdir, sdir, method=method, start_averaging_step=start_averaging_step, x_min=0, x_max=200, rho_min=set_rho_min, rho_max=set_rho_max, mu=set_mu, nr_of_slices=slices, density_avg_exists=density_avg_exists)
                else:
                    func(rdir, sdir, method=method, start_averaging_step=start_averaging_step, x_min=0, x_max=200, rho_min=set_rho_min, rho_max=set_rho_max, nr_of_slices=slices, density_avg_exists=density_avg_exists)

            # Run plot tasks
            for func, rdir, sdir in plot_tasks:
                func(rdir, name="MoveProbgradU", save_dir=sdir)
            
            # Copy run_summary.log from runs_dir to analysis_dir
            run_summary_src = os.path.join(runs_dir, 'run_summary.log')
            run_summary_dst = os.path.join(analysis_dir, 'run_summary.log')
            if os.path.exists(run_summary_src):
                import shutil
                shutil.copy(run_summary_src, run_summary_dst)
                print(f"Copied run_summary.log to {run_summary_dst}")
            else:
                print(f"run_summary.log not found in {runs_dir}")

    # End to measure execution time
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Execution time: {elapsed:.4f} seconds")

