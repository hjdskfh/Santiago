#!/usr/bin/env python3
"""
Enhanced visualization script that automatically parses parameters from .cmd files.

This version integrates with the parse_cmd_params module to automatically extract
all simulation parameters, making it easier to create labeled plots and organize results.

Usage:
    python visualize_with_params.py <runs_directory>
    python visualize_with_params.py runs/run_20250711_123456_uneven-sin_flux/
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Import our parameter parser
from parse_cmd_params import parse_all_cmd_files, extract_parameter_ranges, get_run_by_parameters

# Import existing functions (assuming they exist in your current visualize_simulation.py)
try:
    from postprocessing.helper import create_discrete_colormap, load_occupancy_data, calculate_metrics
    HELPER_FUNCTIONS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import helper functions. Using fallback implementations.")
    HELPER_FUNCTIONS_AVAILABLE = False
    
    # Define minimal fallback functions
    def create_discrete_colormap(max_val):
        import matplotlib.cm as cm
        return cm.get_cmap('viridis')
    
    def load_occupancy_data(folder_path, time_step=-1):
        """
        Fallback function to load occupancy data.
        
        Args:
            folder_path: Path to the folder containing occupancy files
            time_step: Time step to load (-1 for final state)
        """
        occupancy_files = list(Path(folder_path).glob("Occupancy_*.dat"))
        if not occupancy_files:
            return None
        
        if time_step == -1:
            # Get the highest numbered file (final state)
            final_file = sorted(occupancy_files, key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else -1)[-1]
        else:
            # Look for specific time step
            target_file = Path(folder_path) / f"Occupancy_{time_step}.dat"
            if target_file.exists():
                final_file = target_file
            else:
                print(f"Warning: Occupancy_{time_step}.dat not found, using final state")
                final_file = sorted(occupancy_files, key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else -1)[-1]
        
        try:
            return np.loadtxt(final_file)
        except Exception as e:
            print(f"Error loading {final_file}: {e}")
            return None
    
    def calculate_metrics(data):
        return {
            'mean_occupancy': np.mean(data),
            'std_occupancy': np.std(data),
            'max_occupancy': np.max(data),
            'min_occupancy': np.min(data)
        }


class ParameterizedVisualization:
    """
    Enhanced visualization class that uses automatic parameter parsing.
    Completely parameter-agnostic - automatically handles any parameters found in .cmd files.
    """
    
    def __init__(self, runs_directory: str, title_params: List[str] = None, 
                 default_grid_params: Tuple[str, str] = None):
        """
        Initialize the visualization system.
        
        Args:
            runs_directory: Directory containing simulation runs
            title_params: List of parameter names to include in plot titles (auto-detected if None)
            default_grid_params: Tuple of (x_param, y_param) for default grid layout
        """
        self.runs_directory = Path(runs_directory)
        self.all_params = {}
        self.results = []
        self.param_ranges = {}
        
        # Configuration for automatic parameter handling
        self.title_params = title_params  # Will be auto-detected if None
        self.default_grid_params = default_grid_params or ('density', 'tumble-rate')
        
        # Parse all parameters
        self._parse_all_parameters()
        
        # Auto-detect important parameters
        self._detect_parameter_importance()
        
        # Load simulation data
        self._load_simulation_data()
    
    def _parse_all_parameters(self):
        """Parse parameters from all .cmd files in the directory."""
        print("Parsing parameters from .cmd files...")
        
        # Find all subdirectories with .cmd files
        cmd_directories = []
        for item in self.runs_directory.iterdir():
            if item.is_dir() and list(item.glob("*.cmd")):
                cmd_directories.append(item)
        
        # If no subdirectories with .cmd files, check the main directory
        if not cmd_directories and list(self.runs_directory.glob("*.cmd")):
            cmd_directories = [self.runs_directory]
        
        print(f"Found {len(cmd_directories)} directories with .cmd files")
        
        # Parse each directory
        for directory in cmd_directories:
            try:
                dir_params = parse_all_cmd_files(str(directory))
                for run_name, params in dir_params.items():
                    # Use directory name as key for better organization
                    key = f"{directory.name}/{run_name}"
                    params['_directory'] = str(directory)
                    self.all_params[key] = params
            except Exception as e:
                print(f"Warning: Failed to parse {directory}: {e}")
        
        print(f"Successfully parsed parameters for {len(self.all_params)} runs")
        
        # Extract parameter ranges for easy access
        if self.all_params:
            self.param_ranges = extract_parameter_ranges(self.all_params)
            print("\nParameter ranges found:")
            for param, values in sorted(self.param_ranges.items()):
                if len(values) > 1:  # Only show varying parameters
                    print(f"  {param}: {sorted(values)}")
    
    def _detect_parameter_importance(self):
        """
        Automatically detect which parameters are most important for titles and grids.
        """
        if not self.param_ranges:
            return
        
        # Find varying parameters (more than one unique value)
        varying_params = {param: values for param, values in self.param_ranges.items() 
                         if len(values) > 1}
        
        # Auto-detect title parameters if not specified
        if self.title_params is None:
            # Priority order for common parameters
            priority_params = ['density', 'tumble-rate', 'gamma', 'g', 'potential', 
                             'potential-lower', 'potential-upper', 'seed', 'total-time']
            
            self.title_params = []
            for param in priority_params:
                if param in varying_params:
                    self.title_params.append(param)
            
            # Add any other varying numeric parameters
            for param in sorted(varying_params.keys()):
                if param not in self.title_params and param not in priority_params:
                    # Only include if it looks like a numeric parameter
                    sample_value = list(varying_params[param])[0]
                    if isinstance(sample_value, (int, float)):
                        self.title_params.append(param)
        
        # Auto-detect good grid parameters
        numeric_varying = {}
        for param, values in varying_params.items():
            if all(isinstance(v, (int, float)) for v in values):
                numeric_varying[param] = values
        
        if len(numeric_varying) >= 2:
            # Use the two parameters with the most variation
            param_variation = {param: len(values) for param, values in numeric_varying.items()}
            top_params = sorted(param_variation.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Update default grid params if we found better ones
            if top_params[0][0] in ['density', 'tumble-rate'] or top_params[1][0] in ['density', 'tumble-rate']:
                # Keep default if one of the top params is density or tumble-rate
                pass
            else:
                self.default_grid_params = (top_params[1][0], top_params[0][0])  # x, y
        
        print(f"Auto-detected title parameters: {self.title_params}")
        print(f"Default grid parameters: {self.default_grid_params}")
        print(f"Varying parameters found: {list(varying_params.keys())}")
    
    def _load_simulation_data(self):
        """Load occupancy data for all runs."""
        print("Loading simulation data...")
        
        # Try to load a sample first to see what files are available
        sample_directory = None
        for run_key, params in list(self.all_params.items())[:1]:
            sample_directory = params['_directory']
            break
        
        if sample_directory:
            occupancy_files = list(Path(sample_directory).glob("Occupancy_*.dat"))
            print(f"Sample directory {sample_directory} contains {len(occupancy_files)} occupancy files")
            if occupancy_files:
                print(f"  Example files: {[f.name for f in occupancy_files[:3]]}")
        
        successful_loads = 0
        for run_key, params in self.all_params.items():
            directory = params['_directory']
            
            try:
                occupancy_data = self._robust_load_occupancy_data(directory)
                
                if occupancy_data is not None and occupancy_data.size > 0:
                    # Calculate metrics
                    metrics = calculate_metrics(occupancy_data)
                    
                    # Combine everything into a result dict
                    result = {
                        'run_key': run_key,
                        'directory': directory,
                        'occupancy_data': occupancy_data,
                        'params': params,
                        **metrics
                    }
                    
                    # Automatically add ALL parameters as direct attributes for easier access
                    for param_name, param_value in params.items():
                        if not param_name.startswith('_'):  # Skip metadata
                            # Use sanitized parameter names for attributes
                            attr_name = param_name.replace('-', '_')
                            result[attr_name] = param_value
                    
                    self.results.append(result)
                    successful_loads += 1
                    
                    # Only print success for first few to avoid spam
                    if successful_loads <= 3:
                        print(f"✓ Successfully loaded data from {Path(directory).name}")
                else:
                    if successful_loads <= 5:  # Only print first few failures
                        print(f"✗ No valid occupancy data found in {Path(directory).name}")
                    
            except Exception as e:
                if successful_loads <= 5:  # Only print first few failures
                    print(f"✗ Failed to load data from {Path(directory).name}: {e}")
        
        print(f"Successfully loaded data for {len(self.results)} runs out of {len(self.all_params)} total")
        
        if len(self.results) == 0:
            print("\n⚠️  WARNING: No occupancy data could be loaded!")
            print("This might be because:")
            print("  1. The occupancy files have a different naming pattern")
            print("  2. The load_occupancy_data function has different requirements")
            print("  3. The simulation files are incomplete or corrupted")
            print("\nTrying to diagnose the issue...")
            self._diagnose_loading_issue()
    
    def _robust_load_occupancy_data(self, directory: str):
        """
        Robust occupancy data loading that tries multiple approaches.
        
        Args:
            directory: Directory containing occupancy files
            
        Returns:
            Loaded occupancy data or None if failed
        """
        directory_path = Path(directory)
        
        # Method 1: Try the helper function if available
        if HELPER_FUNCTIONS_AVAILABLE:
            try:
                import inspect
                sig = inspect.signature(load_occupancy_data)
                params_info = sig.parameters
                
                if 'time_step' in params_info:
                    # Function expects time_step argument
                    return load_occupancy_data(directory, time_step=-1)
                else:
                    # Function expects only folder_path
                    return load_occupancy_data(directory)
            except Exception as e:
                # If helper function fails, continue to manual loading
                pass
        
        # Method 2: Manual file loading
        try:
            occupancy_files = list(directory_path.glob("Occupancy_*.dat"))
            if not occupancy_files:
                return None
            
            # Get the highest numbered file (final state)
            numbered_files = []
            for f in occupancy_files:
                try:
                    # Extract number from filename like "Occupancy_1000.dat"
                    num_str = f.stem.split('_')[-1]
                    if num_str.isdigit():
                        numbered_files.append((int(num_str), f))
                except:
                    pass
            
            if numbered_files:
                # Sort by number and take the highest
                numbered_files.sort(key=lambda x: x[0])
                final_file = numbered_files[-1][1]
            else:
                # Fallback: take any occupancy file
                final_file = occupancy_files[0]
            
            # Load the file
            data = np.loadtxt(final_file)
            
            # Validate the data
            if data is not None and data.size > 0:
                return data
            else:
                return None
                
        except Exception as e:
            return None
    
    def _diagnose_loading_issue(self):
        """Diagnose why occupancy data loading is failing."""
        print("\n🔍 DIAGNOSTIC INFORMATION:")
        
        # Check a sample directory
        sample_directory = None
        for run_key, params in list(self.all_params.items())[:1]:
            sample_directory = params['_directory']
            break
        
        if not sample_directory:
            print("  No directories found to diagnose")
            return
        
        directory_path = Path(sample_directory)
        print(f"  Examining directory: {directory_path}")
        
        # Check if directory exists
        if not directory_path.exists():
            print(f"  ❌ Directory does not exist!")
            return
        
        # List all files
        all_files = list(directory_path.glob("*"))
        print(f"  📁 Directory contains {len(all_files)} files")
        
        # Look for occupancy files
        occupancy_files = list(directory_path.glob("Occupancy_*.dat"))
        print(f"  📊 Found {len(occupancy_files)} Occupancy_*.dat files")
        
        if occupancy_files:
            print(f"  📋 Occupancy files: {[f.name for f in occupancy_files[:5]]}")
            
            # Try to load one file manually
            sample_file = occupancy_files[0]
            print(f"  🔍 Trying to load {sample_file.name}...")
            
            try:
                data = np.loadtxt(sample_file)
                print(f"  ✅ Successfully loaded: shape={data.shape}, dtype={data.dtype}")
                print(f"  📈 Data range: {np.min(data):.3f} to {np.max(data):.3f}")
            except Exception as e:
                print(f"  ❌ Failed to load {sample_file.name}: {e}")
        else:
            print("  ❌ No Occupancy_*.dat files found!")
            print(f"  📋 Available files: {[f.name for f in all_files[:10]]}")
            
        # Check helper function availability
        print(f"\n  🔧 Helper functions available: {HELPER_FUNCTIONS_AVAILABLE}")
        if HELPER_FUNCTIONS_AVAILABLE:
            try:
                import inspect
                sig = inspect.signature(load_occupancy_data)
                print(f"  📝 load_occupancy_data signature: {sig}")
            except Exception as e:
                print(f"  ❌ Could not inspect load_occupancy_data: {e}")
    
    def create_individual_heatmaps(self):
        """Create individual heatmap files for each run."""
        print("Creating individual heatmaps...")
        
        for result in self.results:
            directory = result['directory']
            params = result['params']
            data = result['occupancy_data']
            
            # Create descriptive title using auto-detected important parameters
            title = self._generate_title_from_params(params)
            
            # Create filename
            output_file = os.path.join(directory, f"heatmap_parsed_params.png")
            
            # Create plot
            self._plot_single_heatmap(data, title, output_file)
        
        print(f"Created {len(self.results)} individual heatmaps")
    
    def _generate_title_from_params(self, params: Dict[str, Any]) -> str:
        """
        Generate a descriptive title from parameters automatically.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Formatted title string
        """
        title_parts = []
        
        # Use auto-detected title parameters
        for param in self.title_params:
            if param in params:
                value = params[param]
                
                # Format based on parameter type and name
                if isinstance(value, bool):
                    if value:  # Only show if True
                        title_parts.append(f"{param}")
                elif isinstance(value, str):
                    if value != 'default':  # Skip default values
                        title_parts.append(f"{param}={value}")
                elif isinstance(value, (int, float)):
                    if value != 0:  # Skip zero values unless they're meaningful
                        # Use Greek letters for common parameters
                        param_symbol = self._get_parameter_symbol(param)
                        if isinstance(value, float):
                            title_parts.append(f"{param_symbol}={value:.3f}")
                        else:
                            title_parts.append(f"{param_symbol}={value}")
        
        return ", ".join(title_parts) if title_parts else "Simulation Result"
    
    def _get_parameter_symbol(self, param_name: str) -> str:
        """
        Get appropriate symbol or abbreviation for parameter names.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Symbol or abbreviated name
        """
        # Greek letters and common abbreviations
        symbol_map = {
            'density': 'ρ',
            'tumble-rate': 'α', 
            'gamma': 'γ',
            'g': 'g',
            'potential': 'pot',
            'potential-lower': 'pot_low',
            'potential-upper': 'pot_up',
            'total-time': 'T',
            'save-interval': 'Δt',
            'seed': 'seed'
        }
        
        return symbol_map.get(param_name, param_name)
    
    def create_parameter_sweep_grid(self, x_param: str = None, y_param: str = None, 
                                   fixed_params: Dict[str, Any] = None):
        """
        Create a grid comparison organized by two varying parameters.
        
        Args:
            x_param: Parameter name for x-axis (columns) - auto-detected if None
            y_param: Parameter name for y-axis (rows) - auto-detected if None
            fixed_params: Dictionary of parameter values to filter by
        """
        # Use auto-detected defaults if not specified
        if x_param is None or y_param is None:
            default_x, default_y = self.default_grid_params
            x_param = x_param or default_x
            y_param = y_param or default_y
        
        print(f"Creating parameter sweep grid: {y_param} vs {x_param}")
        
        # Validate that these parameters exist and vary
        if x_param not in self.param_ranges:
            available = [p for p in self.param_ranges.keys() if len(self.param_ranges[p]) > 1]
            print(f"Warning: Parameter '{x_param}' not found. Available varying parameters: {available}")
            if available:
                x_param = available[0]
                print(f"Using '{x_param}' instead")
            else:
                print("No varying parameters found!")
                return
        
        if y_param not in self.param_ranges:
            available = [p for p in self.param_ranges.keys() if len(self.param_ranges[p]) > 1 and p != x_param]
            print(f"Warning: Parameter '{y_param}' not found. Available varying parameters: {available}")
            if available:
                y_param = available[0]
                print(f"Using '{y_param}' instead")
            else:
                print(f"No second varying parameter found besides '{x_param}'!")
                return
        
        # Filter results by fixed parameters if specified
        filtered_results = self.results
        if fixed_params:
            filtered_results = []
            for result in self.results:
                match = True
                for param, value in fixed_params.items():
                    if result['params'].get(param) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(result)
            print(f"Filtered to {len(filtered_results)} runs matching {fixed_params}")
        
        # Get unique values for grid organization
        x_values = sorted(list(set(r['params'].get(x_param, 0) for r in filtered_results)))
        y_values = sorted(list(set(r['params'].get(y_param, 0) for r in filtered_results)))
        
        n_rows = len(y_values)
        n_cols = len(x_values)
        
        print(f"Grid layout: {n_rows} {y_param} values × {n_cols} {x_param} values")
        print(f"{x_param} values: {x_values}")
        print(f"{y_param} values: {y_values}")
        
        if n_rows == 0 or n_cols == 0:
            print("No data to plot!")
            return
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        # Handle different subplot layouts
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Find global min/max for consistent color scaling
        all_data = [r['occupancy_data'] for r in filtered_results]
        global_min = 0
        global_max = int(np.max([np.max(data) for data in all_data]))
        
        # Create discrete colormap
        discrete_cmap = create_discrete_colormap(global_max)
        
        # Create lookup dictionary
        result_dict = {(r['params'].get(y_param, 0), r['params'].get(x_param, 0)): r 
                      for r in filtered_results}
        
        # Fill the grid
        for row, y_val in enumerate(y_values):
            for col, x_val in enumerate(x_values):
                if (y_val, x_val) in result_dict:
                    result = result_dict[(y_val, x_val)]
                    data = result['occupancy_data']
                    
                    im = axes[row, col].imshow(data, cmap=discrete_cmap, origin='lower', 
                                              aspect='equal', vmin=global_min, vmax=global_max)
                    
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    
                    # Add metrics
                    mean_occ = result['mean_occupancy']
                    std_occ = result['std_occupancy']
                    axes[row, col].text(0.02, 0.98, f'μ={mean_occ:.2f}\nσ={std_occ:.2f}', 
                                       transform=axes[row, col].transAxes, 
                                       verticalalignment='top', fontsize=8,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    axes[row, col].text(0.5, 0.5, 'No Data', 
                                       transform=axes[row, col].transAxes, 
                                       ha='center', va='center')
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
        
        # Add row and column labels with proper symbols
        for row, y_val in enumerate(y_values):
            y_symbol = self._get_parameter_symbol(y_param)
            if isinstance(y_val, float):
                label = f'{y_symbol}={y_val:.3f}'
            else:
                label = f'{y_symbol}={y_val}'
            axes[row, 0].set_ylabel(label, rotation=90, va='center')
        
        for col, x_val in enumerate(x_values):
            x_symbol = self._get_parameter_symbol(x_param)
            if isinstance(x_val, float):
                label = f'{x_symbol}={x_val:.3f}'
            else:
                label = f'{x_symbol}={x_val}'
            axes[0, col].set_title(label)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, shrink=0.6, aspect=20)
        cbar.set_label('Occupancy')
        
        # Create title with proper symbols
        x_symbol = self._get_parameter_symbol(x_param)
        y_symbol = self._get_parameter_symbol(y_param)
        title_parts = [f"Parameter Sweep: {y_symbol} vs {x_symbol}"]
        if fixed_params:
            fixed_parts = []
            for k, v in fixed_params.items():
                k_symbol = self._get_parameter_symbol(k)
                fixed_parts.append(f"{k_symbol}={v}")
            title_parts.append(f"Fixed: {', '.join(fixed_parts)}")
        
        plt.suptitle(" | ".join(title_parts), fontsize=14)
        plt.tight_layout()
        
        # Save plot with sanitized filename
        filename_parts = [f"grid_{y_param.replace('-', '_')}_vs_{x_param.replace('-', '_')}"]
        if fixed_params:
            fixed_str = "_".join([f"{k.replace('-', '_')}{v}" for k, v in fixed_params.items()])
            filename_parts.append(fixed_str)
        
        output_file = self.runs_directory / f"{'_'.join(filename_parts)}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Grid comparison saved to: {output_file}")
        
        plt.show()
    
    def get_available_parameters(self) -> Dict[str, List]:
        """
        Get all available parameters and their possible values.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {param: sorted(list(values)) for param, values in self.param_ranges.items()}
    
    def get_varying_parameters(self) -> List[str]:
        """
        Get list of parameters that vary across runs.
        
        Returns:
            List of parameter names that have more than one unique value
        """
        return [param for param, values in self.param_ranges.items() if len(values) > 1]
    
    def get_numeric_parameters(self) -> List[str]:
        """
        Get list of parameters that are numeric (good for grid axes).
        
        Returns:
            List of numeric parameter names
        """
        numeric_params = []
        for param, values in self.param_ranges.items():
            if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                numeric_params.append(param)
        return numeric_params
    
    def suggest_grid_parameters(self) -> Tuple[str, str]:
        """
        Suggest good parameters for grid visualization.
        
        Returns:
            Tuple of (x_param, y_param) suggestions
        """
        numeric_params = self.get_numeric_parameters()
        
        if len(numeric_params) >= 2:
            # Prefer common parameters if available
            priority = ['density', 'tumble-rate', 'gamma', 'g', 'potential-lower', 'potential-upper']
            
            suggested = []
            for param in priority:
                if param in numeric_params:
                    suggested.append(param)
                    if len(suggested) == 2:
                        break
            
            # Fill with other numeric parameters if needed
            for param in numeric_params:
                if param not in suggested:
                    suggested.append(param)
                    if len(suggested) == 2:
                        break
            
            return tuple(suggested[:2])
        
        return self.default_grid_params
    
    def _plot_single_heatmap(self, data: np.ndarray, title: str, save_path: str):
        """Plot a single heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create colormap
        max_val = int(np.max(data))
        discrete_cmap = create_discrete_colormap(max_val)
        
        im = ax.imshow(data, cmap=discrete_cmap, origin='lower', aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Occupancy')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_parameter_summary(self):
        """Print a comprehensive summary of all parameters found."""
        print("\n" + "="*60)
        print("PARAMETER SUMMARY")
        print("="*60)
        
        if not self.all_params:
            print("No parameters found!")
            return
        
        print(f"Total runs analyzed: {len(self.all_params)}")
        print(f"Runs with valid data: {len(self.results)}")
        
        print(f"\nAuto-detected title parameters: {self.title_params}")
        print(f"Default grid parameters: {self.default_grid_params}")
        
        print("\nAll parameters found:")
        for param, values in sorted(self.param_ranges.items()):
            symbol = self._get_parameter_symbol(param)
            if len(values) > 1:
                print(f"  {param} ({symbol}): {sorted(values)} [{len(values)} values]")
            else:
                print(f"  {param} ({symbol}): {list(values)[0]} [constant]")
        
        # Show varying vs constant parameters
        varying = self.get_varying_parameters()
        constant = [p for p in self.param_ranges.keys() if p not in varying]
        
        print(f"\nVarying parameters ({len(varying)}): {varying}")
        print(f"Constant parameters ({len(constant)}): {constant}")
        
        # Show numeric parameters suitable for grids
        numeric = self.get_numeric_parameters()
        print(f"Numeric varying parameters ({len(numeric)}): {numeric}")
        
        # Show parameter type breakdown
        type_counts = {'numeric': 0, 'string': 0, 'boolean': 0}
        for param, values in self.param_ranges.items():
            sample_value = list(values)[0]
            if isinstance(sample_value, (int, float)):
                type_counts['numeric'] += 1
            elif isinstance(sample_value, bool):
                type_counts['boolean'] += 1
            else:
                type_counts['string'] += 1
        
        print(f"\nParameter types: {type_counts}")
        
        # Automatically group runs by key distinguishing parameters
        self._show_automatic_grouping()
    
    def _show_automatic_grouping(self):
        """Show automatic grouping by most distinguishing parameters."""
        varying_params = self.get_varying_parameters()
        
        if not varying_params:
            print("\nNo varying parameters to group by.")
            return
        
        # Group by most common categorical parameters first
        categorical_params = []
        for param in varying_params:
            sample_value = list(self.param_ranges[param])[0]
            if isinstance(sample_value, str) or isinstance(sample_value, bool):
                categorical_params.append(param)
        
        print(f"\nAutomatic grouping:")
        
        # Group by each categorical parameter
        for param in categorical_params[:3]:  # Limit to first 3
            groups = {}
            for result in self.results:
                value = result['params'].get(param, 'unknown')
                if value not in groups:
                    groups[value] = []
                groups[value].append(result)
            
            print(f"  By {param}:")
            for value, runs in groups.items():
                print(f"    {value}: {len(runs)} runs")
        
        # Show tracking option usage
        tracking_options = [p for p in varying_params if p.startswith('track-')]
        if tracking_options:
            print(f"  Tracking options used:")
            for option in tracking_options:
                enabled = sum(1 for r in self.results if r['params'].get(option, False))
                total = len(self.results)
                print(f"    {option}: {enabled}/{total} runs")


def main():
    """Main function with enhanced command line interface."""
    parser = argparse.ArgumentParser(description="Visualize simulation results with automatic parameter parsing")
    parser.add_argument("runs_directory", help="Directory containing simulation runs")
    parser.add_argument("--individual", action="store_true", help="Create individual heatmaps")
    parser.add_argument("--grid", action="store_true", help="Create parameter sweep grid")
    parser.add_argument("--x-param", help="Parameter for grid x-axis (auto-detected if not specified)")
    parser.add_argument("--y-param", help="Parameter for grid y-axis (auto-detected if not specified)")
    parser.add_argument("--fix", nargs=2, action="append", metavar=("PARAM", "VALUE"), 
                       help="Fix parameter to specific value (can be used multiple times)")
    parser.add_argument("--summary", action="store_true", help="Print parameter summary only")
    parser.add_argument("--list-params", action="store_true", help="List all available parameters and exit")
    parser.add_argument("--suggest-grid", action="store_true", help="Suggest good parameters for grid visualization")
    parser.add_argument("--title-params", nargs="*", help="Specify which parameters to include in titles")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.runs_directory):
        print(f"Error: Directory {args.runs_directory} does not exist")
        return
    
    # Create visualization object
    viz = ParameterizedVisualization(args.runs_directory, title_params=args.title_params)
    
    # Handle parameter listing
    if args.list_params:
        available = viz.get_available_parameters()
        varying = viz.get_varying_parameters()
        numeric = viz.get_numeric_parameters()
        
        print("Available parameters:")
        for param, values in available.items():
            status = ""
            if param in varying:
                status += "[VARYING] "
            if param in numeric:
                status += "[NUMERIC] "
            print(f"  {param}: {values} {status}")
        return
    
    if args.suggest_grid:
        x_param, y_param = viz.suggest_grid_parameters()
        print(f"Suggested grid parameters: x={x_param}, y={y_param}")
        numeric_params = viz.get_numeric_parameters()
        print(f"All numeric parameters available: {numeric_params}")
        return
    
    # Print summary
    viz.print_parameter_summary()
    
    if args.summary:
        return
    
    # Parse fixed parameters with automatic type conversion
    fixed_params = {}
    if args.fix:
        for param, value in args.fix:
            # Try to convert to number if possible
            try:
                if '.' in value or 'e' in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Try boolean
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                # Otherwise keep as string
            fixed_params[param] = value
    
    # Create visualizations
    if args.individual or (not args.grid and not args.individual):
        viz.create_individual_heatmaps()
    
    if args.grid or (not args.grid and not args.individual):
        viz.create_parameter_sweep_grid(args.x_param, args.y_param, fixed_params)


if __name__ == "__main__":
    main()
