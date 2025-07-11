#!/usr/bin/env python3
"""
Simple example showing how to use the CMD parameter parser.

This demonstrates the basic usage patterns for extracting and using
parameters from .cmd files in your analysis scripts.
"""

from parse_cmd_params import parse_cmd_file, parse_all_cmd_files, get_run_by_parameters
import os
import sys


def example_single_cmd_file():
    """Example: Parse a single .cmd file"""
    print("=== Example 1: Parse single .cmd file ===")
    
    # Example assuming you have a .cmd file
    # Replace with actual path to one of your .cmd files
    example_cmd = "runs/run_20250711_123456/d0.5_t0.008_time10000_gamma-0.5_g1.cmd"
    
    if os.path.exists(example_cmd):
        params = parse_cmd_file(example_cmd)
        
        print(f"Parsed parameters from {example_cmd}:")
        for key, value in params.items():
            if not key.startswith('_'):  # Skip metadata
                print(f"  {key}: {value} (type: {type(value).__name__})")
        
        print(f"\nMetadata:")
        if '_metadata' in params:
            for key, value in params['_metadata'].items():
                print(f"  {key}: {value}")
    else:
        print(f"File {example_cmd} not found. Please update the path.")


def example_directory_parsing():
    """Example: Parse all .cmd files in a directory"""
    print("\n=== Example 2: Parse all .cmd files in directory ===")
    
    # Replace with your actual runs directory
    runs_dir = "runs"
    
    if os.path.exists(runs_dir):
        all_params = parse_all_cmd_files(runs_dir)
        
        print(f"Found {len(all_params)} .cmd files in {runs_dir}")
        
        # Show all unique parameter values
        from parse_cmd_params import extract_parameter_ranges
        param_ranges = extract_parameter_ranges(all_params)
        
        print("\nParameter ranges:")
        for param, values in sorted(param_ranges.items()):
            print(f"  {param}: {sorted(values)}")
        
        # Example: Find specific runs
        print("\nExample filtering:")
        high_gamma_runs = get_run_by_parameters(all_params, gamma=-0.5)
        print(f"Runs with gamma=-0.5: {len(high_gamma_runs)}")
        
        flux_tracking_runs = get_run_by_parameters(all_params, **{'track-flux': True})
        print(f"Runs with flux tracking: {len(flux_tracking_runs)}")
        
    else:
        print(f"Directory {runs_dir} not found.")


def example_analysis_workflow():
    """Example: Complete analysis workflow using parsed parameters"""
    print("\n=== Example 3: Analysis workflow ===")
    
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print(f"Directory {runs_dir} not found.")
        return
    
    # Parse all parameters
    all_params = parse_all_cmd_files(runs_dir)
    
    if not all_params:
        print("No .cmd files found.")
        return
    
    print(f"Analyzing {len(all_params)} simulation runs...")
    
    # Group runs by movement probability type
    groups = {}
    for run_name, params in all_params.items():
        potential = params.get('potential', 'default')
        if potential not in groups:
            groups[potential] = []
        groups[potential].append((run_name, params))
    
    print(f"\nRuns grouped by potential type:")
    for potential, runs in groups.items():
        print(f"  {potential}: {len(runs)} runs")
        
        # Show parameter ranges within this group
        group_params = {name: params for name, params in runs}
        from parse_cmd_params import extract_parameter_ranges
        group_ranges = extract_parameter_ranges(group_params)
        
        print(f"    Parameter ranges:")
        for param, values in sorted(group_ranges.items()):
            if len(values) > 1:  # Only show varying parameters
                print(f"      {param}: {sorted(values)}")
    
    # Example: Create parameter combinations for systematic analysis
    print(f"\nExample systematic analysis:")
    
    # Find all unique density and tumble rate combinations
    combinations = set()
    for run_name, params in all_params.items():
        density = params.get('density')
        tumble_rate = params.get('tumble-rate')
        if density is not None and tumble_rate is not None:
            combinations.add((density, tumble_rate))
    
    print(f"Found {len(combinations)} unique (density, tumble_rate) combinations:")
    for density, tumble_rate in sorted(combinations):
        print(f"  density={density}, tumble_rate={tumble_rate}")
        
        # Find all runs with this combination
        matching_runs = get_run_by_parameters(all_params, density=density, **{'tumble-rate': tumble_rate})
        
        # Show what differs between these runs
        if len(matching_runs) > 1:
            print(f"    {len(matching_runs)} runs with this combination - varying parameters:")
            varying_ranges = extract_parameter_ranges(matching_runs)
            for param, values in sorted(varying_ranges.items()):
                if len(values) > 1:
                    print(f"      {param}: {sorted(values)}")


def example_create_analysis_script():
    """Example: Create a template for parameter-driven analysis"""
    print("\n=== Example 4: Template for your analysis ===")
    
    template_code = '''
# Template for parameter-driven analysis
from parse_cmd_params import parse_all_cmd_files, get_run_by_parameters
import numpy as np

def analyze_run_with_params(run_directory, params):
    """Analyze a single run using its parsed parameters."""
    
    # Extract key parameters
    density = params.get('density', 0.0)
    tumble_rate = params.get('tumble-rate', 0.0)
    gamma = params.get('gamma', 0.0)
    potential = params.get('potential', 'default')
    
    print(f"Analyzing: density={density}, tumble_rate={tumble_rate}, potential={potential}")
    
    # Load your data files here
    # occupancy_data = load_occupancy_data(run_directory)
    
    # Adapt analysis based on parameters
    if params.get('track-movement', False):
        print("  Loading movement tracking data...")
        # movement_data = load_movement_data(run_directory)
    
    if params.get('track-flux', False):
        print("  Loading flux tracking data...")
        # flux_data = load_flux_data(run_directory)
    
    # Parameter-specific analysis
    if potential == 'uneven-sin':
        print(f"  Uneven sin potential with gamma={gamma}")
        # Do specific analysis for this potential type
    
    elif potential == 'director-based-sin':
        g = params.get('g', 1.0)
        print(f"  Director-based sin potential with gamma={gamma}, g={g}")
        # Do specific analysis for this potential type
    
    # Return results with parameter context
    return {
        'parameters': params,
        'results': {
            # Your analysis results here
        }
    }

def main():
    # Parse all parameters
    all_params = parse_all_cmd_files("runs/")
    
    # Analyze each run
    all_results = []
    for run_name, params in all_params.items():
        run_directory = params.get('_directory')
        if run_directory:
            result = analyze_run_with_params(run_directory, params)
            all_results.append(result)
    
    # Now you can easily group, filter, and compare results by parameters
    # For example:
    uneven_sin_results = [r for r in all_results 
                         if r['parameters'].get('potential') == 'uneven-sin']
    
    print(f"Found {len(uneven_sin_results)} runs with uneven-sin potential")

if __name__ == "__main__":
    main()
'''
    
    print("Here's a template for using parsed parameters in your analysis:")
    print(template_code)


def main():
    """Run all examples"""
    print("CMD Parameter Parser Examples")
    print("=" * 50)
    
    # Check if we have any runs directory to work with
    possible_dirs = ["runs", "test_debug", "test_debug2"]
    found_dir = None
    
    for directory in possible_dirs:
        if os.path.exists(directory):
            # Check if it has any .cmd files
            import glob
            cmd_files = glob.glob(os.path.join(directory, "**/*.cmd"), recursive=True)
            if cmd_files:
                found_dir = directory
                break
    
    if found_dir:
        print(f"Found simulation data in: {found_dir}")
        print(f"Example .cmd files: {glob.glob(os.path.join(found_dir, '**/*.cmd'), recursive=True)[:3]}")
    else:
        print("No simulation data found. Run some simulations first with:")
        print("  ./run_4_potential.sh --track-flux --track-density")
    
    example_single_cmd_file()
    example_directory_parsing()
    example_analysis_workflow()
    example_create_analysis_script()
    
    print("\n" + "=" * 50)
    print("Quick usage summary:")
    print("1. from parse_cmd_params import parse_cmd_file")
    print("2. params = parse_cmd_file('path/to/run.cmd')")
    print("3. density = params['density']")
    print("4. gamma = params.get('gamma', 0.0)  # with default")
    print("5. Use params in your analysis!")


if __name__ == "__main__":
    main()
