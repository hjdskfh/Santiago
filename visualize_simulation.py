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
    print_moving_particles, visualize_density_evolution_stacked, analyze_density_derivatives_grid, compute_gamma_lambda


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
    print("10. Calculate lambda and gamma for smoothed densities starting from one or several averaging timesteps in a grid (smoothing activated, results printed)")

    while True:
        try:
            mode_choice = input("\nEnter your choice (1-10): ").strip()
            if mode_choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                break
            else:
                print("Please enter a number from 1-10.")
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
        create_parameter_sweep_visualization(runs_dir, number=number, save_dir=analysis_dir)
        print("Visualization complete!")

    elif mode_choice == '2':
        # Create grids for all time steps
        create_parameter_sweep_visualization(runs_dir, process_all_times=True, save_dir=analysis_dir)
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
        save_dir = os.path.join(analysis_dir, "start_heatmaps") if save_choice == 'y' else None

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
        save_dir = os.path.join(analysis_dir, "all_heatmaps") if save_choice == 'y' else None

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
                    save_path = os.path.join(analysis_dir, filename)

            print_single_heatmap(file_path=file_path, save_path=save_path)
            print("Single heatmap complete!")
        else:
            print(f"File not found: {file_path}")

    elif mode_choice == '6':
        # View time evolution of single configuration
        dir_path = input("Enter path to specific configuration folder: ").strip()
        if os.path.exists(dir_path):
            save_choice = input("Save images and animation? (y/n): ").strip().lower()
            save_dir = os.path.join(analysis_dir, "time_evolution") if save_choice == 'y' else None

            show_choice = input("Show individual frames? (y/n): ").strip().lower()
            show_individual = show_choice == 'y'

            visualize_time_evolution(dir_path, save_dir=save_dir, show_individual=show_individual)
            print("Time evolution visualization complete!")
        else:
            print(f"Directory not found: {dir_path}")

    elif mode_choice == '7':
        # Analyze movement statistics
        print("Analyzing movement statistics...")
        analysis_dir = os.path.join(analysis_dir, "moving_particles_statistics")
        print_moving_particles(runs_dir, save_dir=analysis_dir)

    elif mode_choice == '8':
        # Create 2D stacked density evolution (restored option 12)
        save_choice = input("Save stacked density evolution images to file? (y/n): ").strip().lower()
        save_dir = os.path.join(analysis_dir, "density_stacked_evolution") if save_choice == 'y' else None

        show_choice = input("Show individual parameter combinations? (y/n): ").strip().lower()
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


    if mode_choice == '9':
        use_defaults = True  # Change to False if you want to re-enable input
        if use_defaults:
            step_list = list(range(1000, 50001, 7000))  # Example default steps
            kind_of_derivative = 'both'
            smooth_choice = 'y'
            smooth_density = smooth_choice == 'y'
            save_choice = 'y'
            save_dir = os.path.join(analysis_dir, "density_stacked_evolution") if save_choice == 'y' else None
        else:
            while True:
                try:
                    step_input = input("Enter the timesteps to analyze (e.g., 1000, 3000, 5000 or 1000 to 5000 in steps of 1000): ").strip()
                    if 'to' in step_input and 'step' in step_input:
                        # Accept format: 1000 to 5000 in steps of 1000
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
                        # Accept format: 1000 to 5000
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
            # do you want to smooth it?
            smooth_choice = input("Do you want to smooth the density profiles? (y/n): ").strip().lower()
            smooth_density = smooth_choice == 'y'

        save_choice = 'y'
        title_steps = "steps_" + "_".join(map(str, step_list))
        if save_choice == 'y' and smooth_choice == 'y':
            # Create a directory for saving the averaged density profiles
            save_dir = os.path.join(analysis_dir, f"grid_density_average_derivatives_{kind_of_derivative}_{title_steps}_smoothing")
        if save_choice == 'y' and smooth_choice == 'n':
            # Create a directory for saving the averaged density profiles without smoothing
            save_dir = os.path.join(analysis_dir, f"grid_density_average_derivatives_{kind_of_derivative}_{title_steps}")
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        if kind_of_derivative == 'both':
            print("Calculating both Gaussian kernel and finite difference derivatives.")
            analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, save_choice=save_choice, save_dir=save_dir, method='kernel')
            analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, save_choice=save_choice, save_dir=save_dir, method='diff')
        else:
            print(f"Calculating {kind_of_derivative} density derivatives.")
            analyze_density_derivatives_grid(runs_dir, steps_to_include=step_list, smooth=smooth_density, save_choice=save_choice, save_dir=save_dir, method=kind_of_derivative)
  

    if mode_choice == '10':
        # new: I want to calculate lambda derivatives
        use_defaults = True  # Change to False if you want to re-enable input
        if use_defaults:
            start_averaging_step = 20000  # Example default steps
        else:
            while True:
                try:
                    step_input = input("Enter one timestep to analyze (0 or multiples of 1000): ").strip()
                    if step_input == "0":
                        step_list = [0]
                    else:
                        step_list = list(map(int, step_input.split(",")))
                    if all(step % 1000 == 0 for step in step_list):
                        break
                    else:
                        print("Invalid input. Please enter 0 or multiples of 1000.")
                except ValueError:
                    print("Invalid input. Please enter a valid timestep.")
        
        runs_dir_one_particle = input("Enter path to the runs_dir where only a single particle is simulated: ").strip()      

        save_dir = os.path.join(analysis_dir, f"results_gamma_lambda_{start_averaging_step}")
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        compute_gamma_lambda(runs_dir, runs_dir_one_particle, method='diff', start_averaging_step=start_averaging_step, x_min=0, x_max=200)
        # Optionally run flux analysis after derivatives
        # analyze_fluxes_grid(runs_dir, steps_to_include=None, smooth=True, save_choice=False, save_dir=None, method=None)

    # End to measure execution time
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Execution time: {elapsed:.4f} seconds")

    