#!/usr/bin/env python3
"""
Example usage of the CMD parameter parser for simulation analysis.

This demonstrates how to integrate parameter parsing into your analysis workflow.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from parse_cmd_params import parse_cmd_file, parse_all_cmd_files, get_run_by_parameters, extract_parameter_ranges


def analyze_simulation_run(run_directory: str):
    """
    Example function showing how to analyze a single simulation run
    using automatically parsed parameters.
    """
    # Find the .cmd file in the run directory
    cmd_files = list(Path(run_directory).glob("*.cmd"))
    
    if not cmd_files:
        print(f"No .cmd file found in {run_directory}")
        return None
    
    # Parse parameters from the cmd file
    params = parse_cmd_file(str(cmd_files[0]))
    
    print(f"Analyzing run: {run_directory}")
    print(f"Parameters:")
    for key, value in params.items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    # Now you can use these parameters in your analysis
    density = params.get('density', 0.0)
    tumble_rate = params.get('tumble-rate', 0.0)
    gamma = params.get('gamma', 0.0)
    
    # Example: Load data files and analyze based on parameters
    try:
        # Look for occupancy files
        occupancy_files = list(Path(run_directory).glob("Occupancy_*.dat"))
        
        if occupancy_files:
            print(f"Found {len(occupancy_files)} occupancy files")
            
            # Example analysis based on parameters
            if params.get('track-movement', False):
                print("Movement tracking was enabled for this run")
                # Load and analyze movement data
                
            if params.get('track-flux', False):
                print("Flux tracking was enabled for this run")
                # Load and analyze flux data
                
            if params.get('track-density', False):
                print("Density tracking was enabled for this run")
                # Load and analyze density data
                
        return params
        
    except Exception as e:
        print(f"Error analyzing {run_directory}: {e}")
        return None


def compare_parameter_sweep(runs_directory: str):
    """
    Example function showing how to compare results across a parameter sweep.
    """
    print(f"Analyzing parameter sweep in: {runs_directory}")
    
    # Parse all cmd files in the directory
    all_params = parse_all_cmd_files(runs_directory)
    
    if not all_params:
        print("No .cmd files found!")
        return
    
    print(f"Found {len(all_params)} simulation runs")
    
    # Show parameter ranges used in this sweep
    param_ranges = extract_parameter_ranges(all_params)
    print("\nParameter ranges:")
    for param, values in sorted(param_ranges.items()):
        print(f"  {param}: {sorted(values)}")
    
    # Example: Find all runs with specific conditions
    high_density_runs = get_run_by_parameters(all_params, density=0.7)
    if high_density_runs:
        print(f"\nFound {len(high_density_runs)} runs with density=0.7")
    
    # Example: Group runs by movement probability type
    move_prob_groups = {}
    for run_name, params in all_params.items():
        move_prob = params.get('potential', 'default')
        if move_prob not in move_prob_groups:
            move_prob_groups[move_prob] = []
        move_prob_groups[move_prob].append((run_name, params))
    
    print(f"\nRuns grouped by movement probability:")
    for move_prob, runs in move_prob_groups.items():
        print(f"  {move_prob}: {len(runs)} runs")
    
    return all_params


def create_parameter_summary_plot(all_params: dict, output_file: str = None):
    """
    Create a summary plot showing the parameter space explored.
    """
    # Extract numeric parameters for plotting
    densities = []
    tumble_rates = []
    gammas = []
    
    for run_name, params in all_params.items():
        density = params.get('density')
        tumble_rate = params.get('tumble-rate')
        gamma = params.get('gamma')
        
        if density is not None and tumble_rate is not None:
            densities.append(density)
            tumble_rates.append(tumble_rate)
            gammas.append(gamma if gamma is not None else 0)
    
    if not densities:
        print("No numeric parameters found for plotting")
        return
    
    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Density vs Tumble Rate
    scatter1 = ax1.scatter(densities, tumble_rates, c=gammas, cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Density')
    ax1.set_ylabel('Tumble Rate')
    ax1.set_title('Parameter Space: Density vs Tumble Rate')
    plt.colorbar(scatter1, ax=ax1, label='Gamma')
    
    # Plot 2: Parameter distribution
    unique_densities = sorted(set(densities))
    unique_tumble_rates = sorted(set(tumble_rates))
    
    ax2.hist([densities, tumble_rates], bins=10, alpha=0.7, 
             label=['Density', 'Tumble Rate'], density=True)
    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('Normalized Frequency')
    ax2.set_title('Parameter Distributions')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Parameter summary plot saved to: {output_file}")
    else:
        plt.show()


def main():
    """
    Example main function showing different use cases.
    """
    if len(sys.argv) < 2:
        print("Usage: python example_analysis.py <path_to_runs_directory_or_single_run>")
        print("\nExample commands:")
        print("  python example_analysis.py runs/run_20250711_123456/")
        print("  python example_analysis.py runs/run_20250711_123456/d0.5_t0.008_time10000_gamma-0.5_g1/")
        return
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        # Check if this looks like a single run or a collection of runs
        cmd_files = list(Path(path).glob("*.cmd"))
        subdirs_with_cmd = [d for d in Path(path).iterdir() 
                           if d.is_dir() and list(d.glob("*.cmd"))]
        
        if cmd_files and not subdirs_with_cmd:
            # Single run directory
            print("Detected single run directory")
            analyze_simulation_run(path)
            
        elif subdirs_with_cmd:
            # Multiple runs directory
            print("Detected multiple runs directory")
            all_params = compare_parameter_sweep(path)
            
            # Create summary plot
            plot_file = os.path.join(path, "parameter_summary.png")
            create_parameter_summary_plot(all_params, plot_file)
            
        else:
            print(f"No simulation data found in {path}")
    else:
        print(f"Path {path} is not a directory")


if __name__ == "__main__":
    main()
