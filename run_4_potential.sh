#!/bin/bash

# Parameter sweep script for lattice simulation
# Usage: ./run_4_potential.sh [--output-dir DIR] [--move-prob TYPE] [--start-config] [--save-interval N]

# Default feature flags
OUTPUT_DIR=""
MOVE_PROB="default"
USE_START_CONFIG=false  # Default behavior: random configuration for each run
SAVE_INTERVAL=0  # Default: no intermediate saves

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --output-dir requires a directory path"
                exit 1
            fi
            OUTPUT_DIR="$2"
            echo "Using custom output directory: $OUTPUT_DIR"
            shift 2
            ;;
        --move-prob)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --move-prob requires a value (default, uneven-sin)"
                exit 1
            fi
            MOVE_PROB="$2"
            echo "Using movement probability type: $MOVE_PROB"
            shift 2
            ;;
        --start-config)
            USE_START_CONFIG=true
            echo "Creating start configuration"
            shift
            ;;
        --save-interval)
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --save-interval requires a number (e.g., 100)"
                exit 1
            fi
            SAVE_INTERVAL="$2"
            echo "Saving every $SAVE_INTERVAL steps"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_4_potential.sh [--output-dir DIR] [--move-prob TYPE] [--start-config] [--save-interval N]"
            echo "  --output-dir DIR: Use custom output directory instead of 'runs'"
            echo "  --move-prob TYPE: Specify movement probability type (default, uneven-sin)"
            echo "  --start-config: Create start configuration"
            echo "  --save-interval N: Save every N steps for time evolution analysis"
            exit 1
            ;;
    esac
done

# Modify the parameter ranges below as needed

# Create main runs directory if it doesn't exist
BASE_RUNS_DIR="runs"
if [ -n "$OUTPUT_DIR" ]; then
    BASE_RUNS_DIR="$OUTPUT_DIR"
fi

if [ ! -d "$BASE_RUNS_DIR" ]; then
    mkdir -p "$BASE_RUNS_DIR"
    echo "Created '$BASE_RUNS_DIR' directory"
fi

# Create a timestamped runs directory inside the main runs folder
DATETIME=$(date +"%Y%m%d_%H%M%S")
RUNS_DIR="$BASE_RUNS_DIR/run_${DATETIME}"

# Add flags to the run directory name (combinable)
FLAGS=""
if [ "$MOVE_PROB" != "default" ]; then
    FLAGS="${FLAGS}_${MOVE_PROB}"
fi
if [ "$USE_START_CONFIG" = true ]; then
    FLAGS="${FLAGS}_start-config"
fi
if [ -n "$OUTPUT_DIR" ]; then
    FLAGS="${FLAGS}_custom-dir"
fi

# Apply flags to directory name (always include timestamp, add flags if any)
RUNS_DIR="$BASE_RUNS_DIR/run_${DATETIME}${FLAGS}"

echo "Creating timestamped runs directory: $RUNS_DIR"
if [ ! -d "$RUNS_DIR" ]; then
    mkdir "$RUNS_DIR"
    echo "Created '$RUNS_DIR' directory for simulation outputs"
fi

# Compile the program
echo "Compiling lattice2D-Lea-4-potential..."
gcc -o lattice2D-Lea-4-potential lattice2D-Lea-4-potential.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting parameter sweep..."


# ------------ PARAMETER SETTINGS ------------
# Parameter ranges - modify these as needed
densities=(0.7)
#densities=(0.5 0.6 0.7 0.9)  # Added spacing between values
#tumble_rates=(0.001 0.005 0.01 0.05 0.1 0.2)
tumble_rates=(0.001 0.005)
total_time=10000
start_tumble_rate=0.005

# Counter for progress
total_runs=$((${#densities[@]} * ${#tumble_rates[@]}))
current_run=0

# Loop through all parameter combinations
for density in "${densities[@]}"; do
    # Create start configuration only if requested
    if [ "$USE_START_CONFIG" = true ]; then
        start_name="$RUNS_DIR/START_d${density}_t${start_tumble_rate}_time${total_time}"
        echo "[Creating start condition] Running: density=$density, tumble_rate=$start_tumble_rate, run_name=$start_name"
        ./lattice2D-Lea-4-potential $density $start_tumble_rate $total_time $start_name none $MOVE_PROB $SAVE_INTERVAL 
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create start configuration for density $density"
            continue
        fi
    fi
        
    for tumble_rate in "${tumble_rates[@]}"; do
        current_run=$((current_run + 1))
        
        # Create run name and put it in the runs directory
        run_name="$RUNS_DIR/d${density}_t${tumble_rate}_time${total_time}"
        
        echo "[$current_run/$total_runs] Running: density=$density, tumble_rate=$tumble_rate, run_name=$run_name"
        
        # Run the simulation
        if [ "$USE_START_CONFIG" = true ]; then
            # Find the highest numbered occupancy file (excluding initial condition -1)
            highest_file=$(ls "${start_name}"/Occupancy_*.dat 2>/dev/null | grep -v "Occupancy_-1.dat" | sort -V | tail -1)
            if [ -n "$highest_file" ]; then
                echo "Using initial file: $highest_file"
                ./lattice2D-Lea-4-potential $density $tumble_rate $total_time $run_name "$highest_file" $MOVE_PROB $SAVE_INTERVAL
            else
                echo "Warning: No final occupancy files found in $start_name, using random initialization"
                ./lattice2D-Lea-4-potential $density $tumble_rate $total_time $run_name none $MOVE_PROB $SAVE_INTERVAL
            fi
        else
            # Use random initialization (no initial file)
            ./lattice2D-Lea-4-potential $density $tumble_rate $total_time $run_name none $MOVE_PROB $SAVE_INTERVAL
        fi
        
        if [ $? -ne 0 ]; then
            echo "Warning: Simulation failed for $run_name"
        else
            echo "Completed: $run_name"
        fi
    done
done

echo "Parameter sweep completed in directory: $RUNS_DIR"
echo "Configuration used:"
echo "  - Movement probability type: $MOVE_PROB"
echo "  - Start config: $USE_START_CONFIG"
echo "Run 'python visualize_simulation.py $RUNS_DIR' to create visualizations"
