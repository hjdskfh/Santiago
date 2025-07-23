# 🚀 Zero-Effort Parameter Analysis System

**Completely parameter-agnostic visualization and analysis for your simulations!**

This system automatically detects and handles **ANY** parameters from your `.cmd` files. When you add new parameters to `run_4_potential.sh`, the Python analysis code automatically adapts - **no code changes needed!**

## 🎯 Key Features

- **🔄 Automatic Parameter Detection** - Finds all parameters without hardcoding
- **🧠 Smart Auto-Configuration** - Automatically suggests best parameters for titles and grids  
- **🎨 Dynamic Visualization** - Creates labeled plots with proper symbols (ρ, α, γ, etc.)
- **🔧 Zero Maintenance** - Adding new parameters requires no Python code changes
- **📊 Flexible Analysis** - Easy filtering, grouping, and comparison by any parameter combination

## Files Created

- **`parse_cmd_params.py`** - Core parameter parsing utilities
- **`visualize_with_params.py`** - Enhanced visualization with automatic parameter detection
- **`example_analysis.py`** - Example analysis workflow using parsed parameters
- **`example_cmd_usage.py`** - Simple examples showing basic usage

## 🚀 Quick Start

### 1. Run simulations (no changes needed)
```bash
./run_4_potential.sh --move-prob uneven-sin --track-flux --track-density
```

### 2. Automatic analysis (zero configuration)
```python
from visualize_with_params import ParameterizedVisualization

# Automatically detects ALL parameters - no hardcoding!
viz = ParameterizedVisualization("runs/run_20250711_123456/")

# Creates plots with auto-detected parameters
viz.create_individual_heatmaps()  # Automatically titled
viz.create_parameter_sweep_grid()  # Automatically chooses best axes

# Show what was automatically detected
viz.print_parameter_summary()
```

### 3. Adding new parameters (the magic! ✨)
```bash
# 1. Add to run_4_potential.sh - ONLY place you need to edit!
my_temperature=300
cmd="$cmd --my-temperature $my_temperature"

# 2. Run simulation
./run_4_potential.sh --track-flux

# 3. Python automatically handles it - NO CHANGES NEEDED!
viz = ParameterizedVisualization("runs/")  # Automatically detects my-temperature!
```

## 🎛️ Command Line Interface

```bash
# Get intelligent parameter suggestions
python visualize_with_params.py runs/ --suggest-grid
# → Suggested grid parameters: x=density, y=my-temperature

# List all available parameters
python visualize_with_params.py runs/ --list-params

# Auto-create grid with best parameters
python visualize_with_params.py runs/ --grid

# Use specific parameters
python visualize_with_params.py runs/ --x-param density --y-param my-temperature

# Filter by any parameter
python visualize_with_params.py runs/ --fix potential uneven-sin --fix my-temperature 350
```

## ✨ The Magic: Adding New Parameters

**This is the key feature - when you add ANY new parameter to your bash script, the Python code automatically adapts!**

### Step 1: Add to bash script (ONLY place you edit!)
```bash
# In run_4_potential.sh parameter section:
my_temperature=300      # Your new parameter
my_pressure=1.5        # Another new parameter  
my_field_strength=0.8  # Yet another one

# In build_simulation_command function:
cmd="$cmd --my-temperature $my_temperature"
cmd="$cmd --my-pressure $my_pressure"  
cmd="$cmd --my-field-strength $my_field_strength"
```

### Step 2: Run simulation (no changes)
```bash
./run_4_potential.sh --track-flux
```

### Step 3: Analysis automatically works! (no changes)
```python
# Your Python code automatically detects the new parameters!
viz = ParameterizedVisualization("runs/")

# New parameters are automatically:
# ✅ Detected and parsed
# ✅ Available for analysis  
# ✅ Included in titles (if they vary)
# ✅ Available for grid axes (if numeric)
# ✅ Available for filtering

# Use your new parameters immediately:
viz.create_parameter_sweep_grid('density', 'my-temperature')  # Just works!

# Filter by new parameters:
hot_runs = get_run_by_parameters(viz.all_params, **{'my-temperature': 350})

# Access in analysis:
for result in viz.results:
    temp = result['params']['my-temperature']
    pressure = result['params']['my-pressure'] 
    # Analyze based on new parameters...
```

**That's it! No visualization code changes ever needed!** 🎉

## 🎨 Automatic Visualization Features

### Smart Parameter Detection
- **Auto-titles**: Automatically creates descriptive titles using varying parameters
- **Symbol mapping**: Uses proper symbols (ρ for density, α for tumble-rate, γ for gamma)
- **Grid suggestions**: Automatically suggests best parameters for grid visualization
- **Type awareness**: Handles numeric, string, and boolean parameters appropriately

### Flexible Grid Creation
```python
# Automatic grid with best parameters
viz.create_parameter_sweep_grid()

# Specific parameters
viz.create_parameter_sweep_grid('density', 'my-new-parameter')

# With filtering
viz.create_parameter_sweep_grid('density', 'gamma', 
                               fixed_params={'potential': 'uneven-sin'})
```

### Parameter Introspection
```python
# See what's available
available = viz.get_available_parameters()
varying = viz.get_varying_parameters()  
numeric = viz.get_numeric_parameters()

# Get suggestions
x_param, y_param = viz.suggest_grid_parameters()
```

## 🔧 Advanced Usage

### Custom Symbol Mapping
Add symbols for your new parameters once, use everywhere:
```python
def _get_parameter_symbol(self, param_name: str) -> str:
    symbol_map = {
        'density': 'ρ',
        'my-temperature': 'T',     # ← Add your symbols here
        'my-pressure': 'P',        # ← Once and they work everywhere
        # ... existing symbols ...
    }
    return symbol_map.get(param_name, param_name)
```

### Integration with Existing Analysis
```python
from parse_cmd_params import parse_all_cmd_files

def my_analysis_function(runs_directory):
    # Get all parameters automatically
    all_params = parse_all_cmd_files(runs_directory)
    
    for run_name, params in all_params.items():
        # Access ANY parameter (including new ones you add later)
        density = params.get('density', 0.0)
        my_new_param = params.get('my-new-parameter', default_value)
        
        run_dir = params['_directory']
        
        # Your analysis code here...
        result = analyze_simulation(run_dir, density, my_new_param)
        
        # Save with parameter context
        save_results(result, params)
```

## 🎉 Benefits

1. **🔄 Zero-Maintenance Parameter Handling** - Add parameters to bash script, Python automatically adapts
2. **🧠 Intelligent Auto-Detection** - Automatically finds important parameters for titles and grids
3. **🎨 Beautiful Visualizations** - Proper symbols (ρ, α, γ) and smart layout choices  
4. **🔍 Flexible Analysis** - Easy filtering and grouping by any parameter combination
5. **📊 Future-Proof** - Works with any parameters you add in the future
6. **🛡️ Type-Safe** - Automatic type conversion and validation
7. **🎯 Smart Suggestions** - System suggests best parameters for visualization

## 🚀 Examples

Run the demo to see it in action:
```bash
python demo_parameter_agnostic.py
```

See the example files for complete workflows:
- `demo_parameter_agnostic.py` - Shows automatic parameter handling
- `example_analysis.py` - Complete analysis workflow
- `visualize_with_params.py` - Enhanced visualization with auto-detection

## 🔮 The Future is Parameter-Agnostic!

**Your analysis code is now future-proof!** Add any parameters to your bash script and the Python analysis automatically handles them. No more maintaining parameter lists in multiple places!

```bash
# Tomorrow you add these to run_4_potential.sh:
magnetic_field=0.5
temperature=300
wind_speed=10

# Your existing Python analysis code immediately works with them:
viz = ParameterizedVisualization("runs/")  # Auto-detects new parameters!
viz.create_parameter_sweep_grid('density', 'temperature')  # Just works!
```

**Zero effort. Maximum flexibility. Complete automation.** ✨
