import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import ListedColormap
import re
from datetime import datetime

def create_discrete_colormap(max_val):
    return ListedColormap(colors)

def extract_parameters_from_folder(folder_name):
    return None, None, None, None, None

def calculate_metrics(occupancy_data):
    return metrics

def create_individual_heatmaps(results):
    print(f"Created {len(results)} individual heatmaps")

def create_comparison_grid(results, run_date="", number=None):
    # plt.show()  # Comment out to only save, not display

def load_occupancy_data(folder_path, time_step):
        return None, None

def process_folder_for_sweep(folder_path, time_step):
    }

def create_parameter_sweep_visualization(runs_dir='runs', number=1, process_all_times=False):
    create_comparison_grid(results, run_date, number)

def print_single_heatmap(file_path=None, data=None, title=None, save_path=None, show=True):
        print(f"Error creating heatmap: {e}")

def find_files_in_directory(directory, pattern="Occupancy_*.dat", prefix_filter=None):
    return sorted(results)

def print_multiple_heatmaps(runs_dir, time_step=-1, save_dir=None, prefix_filter=None, show_individual=True):
        print_single_heatmap(data=data, title=title, save_path=save_path, show=show_individual)

def visualize_time_evolution(directory_path, save_dir=None, show_individual=False):
    print(f"Time evolution visualization complete! Processed {len(all_data)} time steps.")

def create_evolution_grid(all_data, time_files, dir_name, density, tumble_rate, save_dir=None):
    plt.close()

def print_moving_particles(runs_dir="runs"):
        print("No valid movement data found!")

def create_individual_movement_plot(timesteps, moving_counts, name, density, tumble_rate, gamma, g, run_date):
    print(f"Individual movement plot saved: {filename}")

def create_combined_movement_plots(all_data, run_date=""):
        create_movement_plots_by_tumble_rate(all_data, run_date)

def create_movement_plots_by_tumble_rate(all_data, run_date=""):
        print(f"Movement plot for tumble rate {tumble_rate:.3f} saved: {filename}")

def filter_results_by_parameters(results, density=None, tumble_rate=None, gamma=None, g=None):
    return filtered

def explore_results(results):
            print("Invalid choice")

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
    print("7. Analyze movement statistics")
    
    while True:
        try:
            mode_choice = input("\nEnter your choice (1-7): ").strip()
            if mode_choice in ['1', '2', '3', '4', '5', '6', '7']:
                break
            else:
                print("Please enter 1, 2, 3, 4, 5, 6, or 7.")
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