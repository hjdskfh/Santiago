from postprocessing.all_options import create_individual_heatmaps, visualize_density_evolution_stacked, \
    create_parameter_sweep_visualization, print_multiple_heatmaps, print_single_heatmap, visualize_time_evolution, \
    print_moving_particles, average_density_option_9

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

    # Create a new analysis folder for this run
    analysis_dir = f"analysis/{run_date}" if run_date else "analysis/default"
    os.makedirs(analysis_dir, exist_ok=True)

    print("Starting simulation visualization...")
    print(f"Run date identifier: {run_date}")
    print(f"Analysis output will be saved in: {analysis_dir}")

    # Show options to user
    print("\nChoose visualization mode:")
    print("1. Parameter sweep comparison grid (specific time step)")
    print("2. Parameter sweep grids for ALL time steps")
    print("3. View START configuration heatmaps")
    print("4. View all heatmaps in directory") 
    print("5. View single heatmap from file path")
    print("6. View time evolution of single configuration")
    print("7. Analyze movement statistics")
    print("8. Create 2D stacked density evolution (1D profiles → 2D time evolution)")
    print("9. Average density over x for all runs from a given timestep, with comparison grid")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1-9): ").strip()
            if mode_choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                break
            else:
                print("Please enter a number from 1-9.")
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

    # Option 9: Average density over x for all runs from a given timestep, with comparison grid
    if mode_choice == '9':
        while True:
            try:
                start_input = input("Enter the starting timestep for averaging (e.g., 10000): ").strip()
                start_step = int(start_input)
                break
            except ValueError:
                print("Please enter a valid integer.")

        save_choice = input("Save comparison grid to file? (y/n): ").strip().lower()
        save_dir = os.path.join(analysis_dir, f"density_average_x_start_step_{start_step}") if save_choice == 'y' else None
        average_density_option_9(save_dir=save_dir, save_choice=save_choice, start_step=start_step)
        