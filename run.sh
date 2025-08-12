#!/bin/bash

# Parameter sweep script for lattice simulation
# Usage: ./run_4_potential.sh [--output-dir DIR] [--move-prob TYPE] [--start-config] [--save-interval N|auto] 

# Default feature flags
OUTPUT_DIR=""
MOVE_PROB="default"
USE_START_CONFIG=false  # Default behavior: random configuration for each run
SAVE_INTERVAL="auto"  # Default: auto-calculate from tumble rate
TRACK_MOVEMENT=0  # Default: no movement tracking
TRACK_FLUX=0  # Default: no flux tracking
TRACK_DENSITY=0  # Default: no density tracking
DENSITY_AVERAGING=0  # Default: off
DENSITY_AVERAGING_STEP=""  # Default: unset

# Function to parse command line arguments
parse_arguments() {
    amplitude="0.075"  # Default amplitude, can be overridden by --amplitude
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
                    echo "Error: --move-prob requires a value (default, uneven-sin, director-uneven-sin)"
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
                    echo "Error: --save-interval requires a number or 'auto'"
                    exit 1
                fi
                SAVE_INTERVAL="$2"
                if [ "$SAVE_INTERVAL" != "auto" ]; then
                    echo "Saving every $SAVE_INTERVAL steps"
                else
                    echo "Save interval will be auto-calculated from tumble rate"
                fi
                shift 2
                ;;
            --track-movement)
                TRACK_MOVEMENT=1
                echo "Movement tracking enabled"
                shift
                ;;
            --track-flux)
                TRACK_FLUX=1
                echo "Flux tracking enabled"
                shift
                ;;
            --track-density)
                TRACK_DENSITY=1
                echo "Density tracking enabled"
                shift
                ;;
            --amplitude)
                if [ -z "$2" ] || [[ "$2" == --* ]]; then
                    echo "Error: --amplitude requires a value"
                    exit 1
                fi
                amplitude="$2"
                echo "Amplitude set to $amplitude"
                shift 2
                ;;
            --density-averaging)
                DENSITY_AVERAGING=1
                # Check if next argument is a number (step)
                if [[ -n "$2" && "$2" != --* && "$2" =~ ^[0-9]+$ ]]; then
                    DENSITY_AVERAGING_STEP="$2"
                    echo "Density averaging enabled, starting after step $DENSITY_AVERAGING_STEP"
                    shift 2
                else
                    DENSITY_AVERAGING_STEP=""
                    echo "Density averaging enabled (default step)"
                    shift
                fi
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR           Use custom output directory instead of 'runs', normally not used bc postprocessing assumes 'runs/'"
    echo "  --move-prob TYPE           Specify potential here movement probability type (default (move probability = v0 (default: 0.5)), uneven-sin, director-uneven-sin, director-symmetric-sin), normally director-uneven-sin used to get uneven density direction based potential"
    echo "  --start-config             Create start configuration and use it for all runs, otherwise random initialization for each run (default: random initialization)"
    echo "  --save-interval N          Save every N steps (default: Save every 1/tumble_rate)"
    echo "  --track-movement           Enable movement tracking at save intervals, so movement_stats.txt is created (needed for visualizing observable moving particles option 7 in visualize.py)"
    echo "  --track-flux               Enable flux tracking (raw accumulated values) (needed to calculate flux in postprocessing option 10)"
    echo "  --track-density            Enable density tracking (needed for postprocessing options 8,9,10)"
    echo "  --density-averaging [STEP] Enable density averaging after given step (default: 10000 if not specified) --> makes computation faster"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Note: amplitude is set with --amplitude flag in terminal.sh, default is 0.075. This is the amplitude of the move probability from the potential used in the simulation. This is added to v0."
    echo "Variables set following lines 276:"
    echo "  densities: sets the densities to be used in the simulation for every run"
    echo "  tumble_rates: sets the tumble rates to be used in the simulation. Use only value for all runs if you want to calculate lambda and gamma over rho and the amplitude of g"
    echo "  total_time: sets the total time steps for the simulation. 4 mio steps take about 2 hours per setting"
    echo "  start_tumble_rate: sets the start tumble rate for the simulation, if there is a start configuration, this is used to create the start configuration"
    echo "  gamma: sets the gamma parameter for the simulation, this is used for the uneven-sin potential"
    echo "  v0: default move probability. This is the value that the move probability varies around. Needs to be between 0 and 1."
    echo "  seed: sets the seed parameter for the simulation"
}

# Function to calculate save interval from tumble rate
calculate_save_interval() {
    local tumble_rate="$1"
    # Use awk instead of bc for better portability
    echo "$tumble_rate" | awk '{printf "%.0f", 1/$1}'
}

# Function to build simulation command - automatically uses all defined parameters
build_simulation_command() {
    local density="$1"
    local tumble_rate="$2"
    local total_time="$3"
    local run_name="$4"
    local initial_file="$5"
    local current_save_interval="$6"
    
    # Build basic command with required parameters
    local cmd="./lattice2D --density $density --tumble-rate $tumble_rate --total-time $total_time --run-name $run_name"
    
    # Add initial file if not "none"
    if [ "$initial_file" != "none" ]; then
        cmd="$cmd --initial-file $initial_file"
    fi
    
    # Add save interval
    cmd="$cmd --save-interval $current_save_interval"
    
    # ============ PARAMETER SECTION - ONLY PLACE TO CHANGE ============
    # Add your parameters here using the pattern: cmd="$cmd --parameter-name $variable"
    # The C code will automatically recognize any --parameter-name you add
    
    cmd="$cmd --potential $MOVE_PROB"
    cmd="$cmd --gamma $gamma"
    cmd="$cmd --v0 $v0"
    cmd="$cmd --amplitude $amplitude"
    cmd="$cmd --seed $seed"
    
    # Add flags
    if [ "$TRACK_MOVEMENT" = "1" ]; then
        cmd="$cmd --track-movement"
    fi
    if [ "$TRACK_FLUX" = "1" ]; then
        cmd="$cmd --track-flux"
    fi
    if [ "$TRACK_DENSITY" = "1" ]; then
        cmd="$cmd --track-density"
    fi
    
    # Density averaging flag (with optional step)
    if [ "$DENSITY_AVERAGING" = "1" ]; then
        if [ -n "$DENSITY_AVERAGING_STEP" ]; then
            cmd="$cmd --density-averaging $DENSITY_AVERAGING_STEP"
        else
            cmd="$cmd --density-averaging"
        fi
    fi
    # To add a new parameter, just add one line here:
    # cmd="$cmd --new-parameter $new_variable"
    # That's it! No other changes needed anywhere else.
    
    echo "$cmd"
}

# Function to log command for reproducibility
log_command() {
    local cmd="$1"
    local run_name="$2"
    local log_file="$run_name.cmd"
    
    echo "# Command executed on $(date)" > "$log_file"
    echo "# Working directory: $(pwd)" >> "$log_file"
    echo "# Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not available')" >> "$log_file"
    echo "$cmd" >> "$log_file"
    echo "Command logged to: $log_file"
}

# Parse command line arguments
parse_arguments "$@"

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
if [ "$TRACK_FLUX" = "1" ]; then
    FLAGS="${FLAGS}_flux"
fi
if [ "$TRACK_DENSITY" = "1" ]; then
    FLAGS="${FLAGS}_density"
fi
if [ "$DENSITY_AVERAGING" = "1" ]; then
    FLAGS="${FLAGS}_densavg"
    if [ -n "$DENSITY_AVERAGING_STEP" ]; then
        FLAGS="${FLAGS}${DENSITY_AVERAGING_STEP}"
    fi
fi
FLAGS="${FLAGS}_amplitude${amplitude}"

# Apply flags to directory name (always include timestamp, add flags if any)
RUNS_DIR="$BASE_RUNS_DIR/run_${DATETIME}${FLAGS}"

echo "Creating timestamped runs directory: $RUNS_DIR"
if [ ! -d "$RUNS_DIR" ]; then
    mkdir "$RUNS_DIR"
    echo "Created '$RUNS_DIR' directory for simulation outputs"
fi

# Create master log file for the entire run
MASTER_LOG="$RUNS_DIR/run_summary.log"
echo "# Parameter sweep run started on $(date)" > "$MASTER_LOG"
echo "# Working directory: $(pwd)" >> "$MASTER_LOG"
echo "# Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not available')" >> "$MASTER_LOG"
echo "# Command line: $0 $@" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Compile the program with optimization flags
echo "Compiling lattice2D with optimizations..."
gcc -O3 -march=native -ffast-math -funroll-loops -o lattice2D lattice2D.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting parameter sweep..."


# ============ PARAMETER SETTINGS - SINGLE SOURCE OF TRUTH ============
# Parameter ranges - modify these as needed
densities=(0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9)
tumble_rates=(0.11)./
total_time=16000000
start_tumble_rate=0.005

gamma=-0.5              # Gamma parameter for sin potentials
v0=0.5
seed=837437             # Random seed (example of new parameter)
# amplitude is set in parse_arguments, default 0.075, can be overridden by --amplitude

# To add a new parameter:
# 1. Add: new_parameter=value    # here
# 2. Add: cmd="$cmd --new-parameter $new_parameter"    # in build_simulation_command

# Counter for progress
total_runs=$((${#densities[@]} * ${#tumble_rates[@]}))
current_run=0

# Loop through all parameter combinations
for density in "${densities[@]}"; do
    # Create start configuration only if requested
    if [ "$USE_START_CONFIG" = true ]; then
        start_name="$RUNS_DIR/START_d${density}_t${start_tumble_rate}_time${total_time}"
        echo "[Creating start condition] Running: density=$density, tumble_rate=$start_tumble_rate, run_name=$start_name"
        
        # Calculate save interval for start configuration
        start_save_interval="$SAVE_INTERVAL"
        if [ "$SAVE_INTERVAL" = "auto" ]; then
            start_save_interval=$(calculate_save_interval "$start_tumble_rate")
        fi
        
        # change potential used for start configuration
        ORIGINAL_MOVE_PROB="$MOVE_PROB"
        MOVE_PROB="uneven-sin"

        # Build command for start configuration
        start_cmd=$(build_simulation_command "$density" "$start_tumble_rate" "$total_time" "$start_name" "none" "$start_save_interval")
        
        # Restore original MOVE_PROB for main runs
        MOVE_PROB="$ORIGINAL_MOVE_PROB"
        
        # Log the start command
        log_command "$start_cmd" "$start_name"
        echo "START: $start_cmd" >> "$MASTER_LOG"
        
        eval "$start_cmd"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create start configuration for density $density"
            echo "FAILED START: $start_cmd" >> "$MASTER_LOG"
            continue
        else
            echo "SUCCESS START: $start_cmd" >> "$MASTER_LOG"
        fi
    fi
        
    for tumble_rate in "${tumble_rates[@]}"; do
        current_run=$((current_run + 1))
        
        # Create run name and put it in the runs directory
        run_name="$RUNS_DIR/d${density}_t${tumble_rate}_time${total_time}_gamma${gamma}"
        
        echo "[$current_run/$total_runs] Running: density=$density, tumble_rate=$tumble_rate, run_name=$run_name"
        
        # Determine initial file
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
        
        # Calculate save interval for this tumble rate
        current_save_interval="$SAVE_INTERVAL"
        if [ "$SAVE_INTERVAL" = "auto" ]; then
            current_save_interval=$(calculate_save_interval "$tumble_rate")
            echo "Auto-calculated save interval: $current_save_interval (from tumble rate $tumble_rate)"
        fi
        
        # Build the simulation command
        cmd=$(build_simulation_command "$density" "$tumble_rate" "$total_time" "$run_name" "$initial_file" "$current_save_interval")
        
        # Log the command for reproducibility
        log_command "$cmd" "$run_name"
        
        # Log to master log
        echo "[$current_run/$total_runs] $cmd" >> "$MASTER_LOG"
        
        # Execute the command
        eval "$cmd"
        
        if [ $? -ne 0 ]; then
            echo "Warning: Simulation failed for $run_name"
            echo "FAILED: [$current_run/$total_runs] $cmd" >> "$MASTER_LOG"
        else
            echo "Completed: $run_name"
            echo "SUCCESS: [$current_run/$total_runs] $cmd" >> "$MASTER_LOG"
        fi
    done
done

# -------- PRINT SUMMARY AND CREATE LOGS --------

# Print final summary
echo "Parameter sweep completed in directory: $RUNS_DIR"
echo "Configuration used:"
echo "  - Movement probability type: $MOVE_PROB"
echo "  - Start config: $USE_START_CONFIG"
echo "  - Save interval: $SAVE_INTERVAL"
echo "  - Movement tracking: $TRACK_MOVEMENT"
echo "  - Flux tracking: $TRACK_FLUX"
echo "  - Density tracking: $TRACK_DENSITY"
if [ "$MOVE_PROB" = "uneven-sin" ] || [ "$MOVE_PROB" = "director-uneven-sin" ] || [ "$MOVE_PROB" = "director-symmetric-sin" ]; then
    echo "  - Gamma parameter: $gamma"
fi


# Write final summary to master log
echo "" >> "$MASTER_LOG"
echo "# Parameter sweep completed on $(date)" >> "$MASTER_LOG"
echo "# Total runs: $total_runs" >> "$MASTER_LOG"
echo "# Configuration:" >> "$MASTER_LOG"
echo "#   - Movement probability type: $MOVE_PROB" >> "$MASTER_LOG"
echo "#   - Start config: $USE_START_CONFIG" >> "$MASTER_LOG"
echo "#   - Save interval: $SAVE_INTERVAL" >> "$MASTER_LOG"
echo "#   - Movement tracking: $TRACK_MOVEMENT" >> "$MASTER_LOG"
echo "#   - Flux tracking: $TRACK_FLUX" >> "$MASTER_LOG"
echo "#   - Density tracking: $TRACK_DENSITY" >> "$MASTER_LOG"
if [ "$MOVE_PROB" = "uneven-sin" ] || [ "$MOVE_PROB" = "director-uneven-sin" ]; then
    echo "#   - Gamma parameter: $gamma" >> "$MASTER_LOG"
fi
if [ "$MOVE_PROB" = "director-uneven-sin" ] || [ "$MOVE_PROB" = "director-symmetric-sin" ]; then
    echo "#   - G parameter: $g" >> "$MASTER_LOG"
fi

echo "Master log written to: $MASTER_LOG"
echo "Individual command logs: $RUNS_DIR/*.cmd"
echo "Run 'python visualize_simulation.py $RUNS_DIR' to create visualizations"
