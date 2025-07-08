#!/bin/bash

# Parameter sweep script for lattice simulation
# Usage: ./run_4_potential.sh [--output-dir DIR] [--move-prob TYPE] [--start-config] [--save-interval N] 

# Default feature flags
OUTPUT_DIR=""
MOVE_PROB="default"
USE_START_CONFIG=false  # Default behavior: random configuration for each run
SAVE_INTERVAL=0  # Default: no intermediate saves
TRACK_MOVEMENT=0  # Default: no movement tracking
CALCULATE_DENSITY_AND_FLUX=0  # Default: no density and flux calculation

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
        --track-movement)
            TRACK_MOVEMENT=1
            echo "Movement tracking enabled"
            shift
            ;;
        --create-density-and-flux)
            CALCULATE_DENSITY_AND_FLUX=1
            if [ -z "$2" ] || [[ "$2" == --* ]]; then
                echo "Error: --calculate-density-and-flux requires a start step number"
                exit 1
            fi
            START_STEP_FOR_CALC="$2"
            echo "Density and flux calculation enabled with start step $START_STEP_FOR_CALC"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_4_potential.sh [--output-dir DIR] [--move-prob TYPE] [--start-config] [--save-interval N] [--track-movement]"
            echo "  --output-dir DIR: Use custom output directory instead of 'runs'"
            echo "  --move-prob TYPE: Specify movement probability type (default, uneven-sin, director-based-sin)"
            echo "  --start-config: Create start configuration"
            echo "  --save-interval N: Save every N steps for time evolution analysis"
            echo "  --track-movement: Enable movement tracking at save intervals"
            echo "  --create-density-and-flux: Enable density and flux calculation"
            echo "Note: Gamma and G parameters are set in the script's parameter section"
        
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
    FLAGS="${FLAGS}_start"
fi
if [ "$TRACK_MOVEMENT" = "1" ]; then
    FLAGS="${FLAGS}_track"
fi
if [ "$CALCULATE_DENSITY_AND_FLUX" = "1" ]; then
    FLAGS="${FLAGS}_calc${START_STEP_FOR_CALC}"
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
densities=(0.5 0.6 0.7)
tumble_rates=(0.001 0.005 0.01 0.05 0.1)
total_time=10000
start_tumble_rate=0.005

# Potential-specific parameters (modify as needed)
gamma=-0.5  # For uneven-sin and director-based-sin potentials
g=1       # For director-based-sin potential only

# Counter for progress
total_runs=$((${#densities[@]} * ${#tumble_rates[@]}))
current_run=0

# Loop through all parameter combinations
for density in "${densities[@]}"; do
    # Create start configuration only if requested
    if [ "$USE_START_CONFIG" = true ]; then
        start_name="$RUNS_DIR/START_d${density}_t${start_tumble_rate}_time${total_time}"
        echo "[Creating start condition] Running: density=$density, tumble_rate=$start_tumble_rate, run_name=$start_name"
        
        # Build command for start configuration using new named parameter format
        start_cmd="./lattice2D-Lea-4-potential --density $density --tumble-rate $start_tumble_rate --total-time $total_time --run-name $start_name --potential $MOVE_PROB"
        
        # Add potential-specific parameters for start configuration
        if [ "$MOVE_PROB" = "uneven-sin" ]; then
            start_cmd="$start_cmd --gamma $gamma"
        elif [ "$MOVE_PROB" = "director-based-sin" ]; then
            start_cmd="$start_cmd --gamma $gamma --g $g"
        fi
        
        $start_cmd 
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create start configuration for density $density"
            continue
        fi
    fi
        
    for tumble_rate in "${tumble_rates[@]}"; do
        current_run=$((current_run + 1))
        
        # Create run name and put it in the runs directory
        run_name="$RUNS_DIR/d${density}_t${tumble_rate}_time${total_time}_gamma${gamma}_g${g}"
        
        echo "[$current_run/$total_runs] Running: density=$density, tumble_rate=$tumble_rate, run_name=$run_name"
        
        # Run the simulation
        initial_file=none
        if [ "$USE_START_CONFIG" = true ]; then
            highest_file=$(ls "${start_name}"/Occupancy_*.dat 2>/dev/null | grep -v "Occupancy_-1.dat" | sort -V | tail -1)
            if [ -n "$highest_file" ]; then
                initial_file="$highest_file"
                echo "Using initial file: $highest_file"
            else
                echo "Warning: No final occupancy files found in $start_name, using random initialization"
            fi
        fi
        
        # Run the simulation - build command using new named parameter format
        cmd="./lattice2D-Lea-4-potential --density $density --tumble-rate $tumble_rate --total-time $total_time --run-name $run_name"
        
        # Add initial file if not "none"
        if [ "$initial_file" != "none" ]; then
            cmd="$cmd --initial-file $initial_file"
        fi
        
        # Add potential type
        cmd="$cmd --potential $MOVE_PROB"
        
        # Add save interval
        cmd="$cmd --save-interval $SAVE_INTERVAL"
        
        # Add movement tracking
        if [ "$TRACK_MOVEMENT" = "1" ]; then
            cmd="$cmd --track-movement"
        fi
        
        # Add density and flux calculation flags
        if [ "$CALCULATE_DENSITY_AND_FLUX" = "1" ]; then
            cmd="$cmd --track-density --track-flux"
            
            # Calculate the save interval for this tumble rate (if not explicitly set)
            current_save_interval="$SAVE_INTERVAL"
            if [ "$SAVE_INTERVAL" = "0" ]; then
                # Default: use 1/tumbling_rate
                current_save_interval=$(echo "scale=0; 1 / $tumble_rate" | bc)
            fi
            
            # Adjust start step to align with save intervals if needed
            adjusted_start_step="$START_STEP_FOR_CALC"
            if [ "$current_save_interval" -gt 0 ]; then
                remainder=$(($START_STEP_FOR_CALC % $current_save_interval))
                if [ "$remainder" -ne 0 ]; then
                    adjusted_start_step=$(($START_STEP_FOR_CALC + $current_save_interval - $remainder))
                    echo "Adjusting start step from $START_STEP_FOR_CALC to $adjusted_start_step to align with save interval $current_save_interval"
                fi
            fi
            
            cmd="$cmd --start-calc-step $adjusted_start_step"
        fi
        
        # Add potential-specific parameters
        if [ "$MOVE_PROB" = "uneven-sin" ]; then
            cmd="$cmd --gamma $gamma"
        elif [ "$MOVE_PROB" = "director-based-sin" ]; then
            cmd="$cmd --gamma $gamma --g $g"
        fi
        # No extra parameters needed for "default" type
        
        eval "$cmd"
        
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
echo "  - Save interval: $SAVE_INTERVAL"
echo "  - Movement tracking: $TRACK_MOVEMENT"
echo "  - calculate density and flux: $CALCULATE_DENSITY_AND_FLUX from step $START_STEP_FOR_CALC"
if [ "$MOVE_PROB" = "uneven-sin" ] || [ "$MOVE_PROB" = "director-based-sin" ]; then
    echo "  - Gamma parameter: $gamma"
fi
if [ "$MOVE_PROB" = "director-based-sin" ]; then
    echo "  - G parameter: $g"
fi
echo "Run 'python visualize_simulation.py $RUNS_DIR' to create visualizations"
