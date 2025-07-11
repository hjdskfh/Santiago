#!/usr/bin/env python3
"""
Demo: Adding New Parameters - No Code Changes Needed!

This script demonstrates how the new parameter-agnostic visualization system
automatically handles any new parameters you add to run_4_potential.sh.

NO CODE CHANGES ARE NEEDED when you add new parameters!
"""

from visualize_with_params import ParameterizedVisualization
import os


def demo_automatic_parameter_detection():
    """Demo showing automatic parameter detection."""
    print("="*60)
    print("DEMO: Automatic Parameter Detection")
    print("="*60)
    
    # This works with ANY runs directory - no hardcoded parameters!
    runs_dir = "runs"  # Change this to your actual runs directory
    
    if not os.path.exists(runs_dir):
        print(f"Please run some simulations first or change runs_dir to an existing directory")
        return
    
    # Initialize the visualization system
    viz = ParameterizedVisualization(runs_dir)
    
    # Show what was automatically detected
    print("\n🔍 AUTOMATIC DETECTION RESULTS:")
    print("-" * 40)
    
    available_params = viz.get_available_parameters()
    varying_params = viz.get_varying_parameters()
    numeric_params = viz.get_numeric_parameters()
    
    print(f"📊 Total parameters found: {len(available_params)}")
    print(f"🔄 Varying parameters: {len(varying_params)}")
    print(f"🔢 Numeric parameters: {len(numeric_params)}")
    
    print(f"\n📋 All varying parameters:")
    for param in varying_params:
        values = available_params[param]
        print(f"   {param}: {values}")
    
    # Show automatic grid suggestions
    x_param, y_param = viz.suggest_grid_parameters()
    print(f"\n🎯 Suggested grid axes: x={x_param}, y={y_param}")
    
    print(f"\n🏷️  Auto-detected title parameters: {viz.title_params}")
    
    # Show how to create visualizations without specifying parameters
    print(f"\n🎨 Creating visualizations...")
    print(f"   Individual heatmaps: automatically titled with relevant parameters")
    print(f"   Grid comparison: automatically uses best varying parameters")
    
    # Uncomment these to actually create the plots:
    # viz.create_individual_heatmaps()
    # viz.create_parameter_sweep_grid()  # Uses auto-detected parameters!


def demo_adding_new_parameter():
    """Demo showing what happens when you add a new parameter."""
    print("\n\n" + "="*60)
    print("DEMO: Adding New Parameters")
    print("="*60)
    
    print("""
🚀 TO ADD A NEW PARAMETER TO YOUR ANALYSIS:

1. Add to run_4_potential.sh:
   ```bash
   # In parameter section:
   new_temperature=300    # Your new parameter
   
   # In build_simulation_command function:
   cmd="$cmd --new-temperature $new_temperature"
   ```

2. Run your simulation:
   ```bash
   ./run_4_potential.sh --track-flux
   ```

3. Use in Python - NO CHANGES NEEDED:
   ```python
   from visualize_with_params import ParameterizedVisualization
   
   # Automatically detects ALL parameters including new ones!
   viz = ParameterizedVisualization("runs/run_20250711_123456/")
   
   # Your new parameter is automatically:
   # ✅ Parsed from .cmd files
   # ✅ Available in parameter lists
   # ✅ Included in titles if it varies
   # ✅ Available for grid axes if numeric
   # ✅ Available for filtering
   
   # Access your new parameter:
   for result in viz.results:
       temperature = result['params'].get('new-temperature', 0)
       print(f"Temperature: {temperature}")
   
   # Use in grid visualization (if it varies):
   viz.create_parameter_sweep_grid('density', 'new-temperature')
   
   # Filter by your new parameter:
   from parse_cmd_params import get_run_by_parameters
   hot_runs = get_run_by_parameters(viz.all_params, **{'new-temperature': 350})
   ```

🎉 That's it! No visualization code changes needed!
""")


def demo_flexible_usage():
    """Demo showing flexible usage patterns."""
    print("\n\n" + "="*60)
    print("DEMO: Flexible Usage Patterns")
    print("="*60)
    
    print("""
🔧 FLEXIBLE USAGE EXAMPLES:

# 1. Automatic everything:
viz = ParameterizedVisualization("runs/")
viz.create_parameter_sweep_grid()  # Uses best detected parameters

# 2. Custom title parameters:
viz = ParameterizedVisualization("runs/", title_params=['density', 'my-new-param'])

# 3. Get available parameters programmatically:
available = viz.get_available_parameters()
varying = viz.get_varying_parameters()
numeric = viz.get_numeric_parameters()

# 4. Smart parameter suggestions:
x_param, y_param = viz.suggest_grid_parameters()
viz.create_parameter_sweep_grid(x_param, y_param)

# 5. Filter and visualize:
viz.create_parameter_sweep_grid('density', 'tumble-rate', 
                               fixed_params={'potential': 'uneven-sin'})

# 6. Command line usage:
# python visualize_with_params.py runs/ --list-params     # See all parameters
# python visualize_with_params.py runs/ --suggest-grid    # Get suggestions
# python visualize_with_params.py runs/ --grid            # Auto-detect axes
# python visualize_with_params.py runs/ --x-param density --y-param my-new-param
""")


def demo_parameter_symbols():
    """Demo showing automatic parameter symbol detection."""
    print("\n\n" + "="*60)
    print("DEMO: Automatic Parameter Symbols")
    print("="*60)
    
    print("""
🔤 AUTOMATIC SYMBOL MAPPING:

The system automatically uses appropriate symbols/abbreviations:

Parameter Name       → Symbol Used in Plots
density             → ρ
tumble-rate         → α  
gamma               → γ
potential           → pot
total-time          → T
save-interval       → Δt
my-new-parameter    → my-new-parameter (uses name if no symbol defined)

To add symbols for your new parameters, just edit the _get_parameter_symbol() 
method once and it works everywhere:

```python
def _get_parameter_symbol(self, param_name: str) -> str:
    symbol_map = {
        'density': 'ρ',
        'tumble-rate': 'α', 
        'gamma': 'γ',
        'my-temperature': 'T',    # ← Add your new parameter here
        'my-pressure': 'P',       # ← And here
        # ... existing symbols ...
    }
    return symbol_map.get(param_name, param_name)
```

Then ALL plots, titles, and labels automatically use your symbols! 🎨
""")


def main():
    """Run all demos."""
    print("🚀 PARAMETER-AGNOSTIC VISUALIZATION DEMO")
    print("This shows how the system automatically handles ANY parameters!")
    
    demo_automatic_parameter_detection()
    demo_adding_new_parameter()
    demo_flexible_usage()
    demo_parameter_symbols()
    
    print("\n\n" + "="*60)
    print("🎉 SUMMARY: ZERO-EFFORT PARAMETER HANDLING")
    print("="*60)
    print("""
✅ Automatically detects ALL parameters from .cmd files
✅ No hardcoded parameter lists to maintain
✅ Smart auto-detection of important parameters for titles
✅ Automatic suggestions for grid visualization
✅ Type-aware handling (numeric, string, boolean)
✅ Flexible command-line interface
✅ Easy filtering and grouping by any parameter combination
✅ Extensible symbol mapping for pretty plots

🔮 ADDING NEW PARAMETERS:
1. Add to bash script ← Only place you need to make changes!
2. Run simulation
3. Python code automatically handles it ← No changes needed!

🚀 Your analysis code is now future-proof!
""")


if __name__ == "__main__":
    main()
