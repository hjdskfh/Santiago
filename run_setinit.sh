#!/bin/bash

# Parameter sweep script for lattice simulation
# Modify the parameter ranges below as needed

# Create main runs directory if it doesn't exist
if [ ! -d "runs" ]; then
    mkdir "runs"
    echo "Created 'runs' directory"
fi

# Create a timestamped runs directory inside the main runs folder
DATETIME=$(date +"%Y%m%d_%H%M%S")
RUNS_DIR="runs/run_${DATETIME}"

echo "Creating timestamped runs directory: $RUNS_DIR"
if [ ! -d "$RUNS_DIR" ]; then
    mkdir "$RUNS_DIR"
    echo "Created '$RUNS_DIR' directory for simulation outputs"
fi

# Compile the program
echo "Compiling lattice2D-Lea-setinit..."
gcc -o lattice2D-Lea-setinit lattice2D-Lea-setinit.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting parameter sweep..."

# Parameter ranges - modify these as needed
#densities=(0.7)
densities=(0.5 0.6 0.7 0.9)  # Added spacing between values
tumble_rates=(0.001 0.005 0.01 0.05 0.1 0.2)
total_time=10000
start_tumble_rate=0.005
start_total_time=1000

# Counter for progress
total_runs=$((${#densities[@]} * ${#tumble_rates[@]}))
current_run=0

# Loop through all parameter combinations
for density in "${densities[@]}"; do
    # create new startposition and leave out the initialisation file

    start_name="$RUNS_DIR/d${start_density}_t${start_tumble_rate}_time${start_total_time}"
    start_density=$density  # Use the current density for the start condition
    echo "[Creating start condition] Running: density=$start_density, tumble_rate=$start_tumble_rate, run_name=$start_name"
    ./lattice2D-Lea-setinit $start_density $start_tumble_rate $start_total_time $start_name 
        
    for tumble_rate in "${tumble_rates[@]}"; do
        current_run=$((current_run + 1))
        
        # Create run name and put it in the runs directory
        run_name="$RUNS_DIR/d${density}_t${tumble_rate}_time${total_time}"
        
        echo "[$current_run/$total_runs] Running: density=$density, tumble_rate=$tumble_rate, run_name=$run_name"
        
        # Run the simulation (using the highest numbered occupancy file from start condition, excluding initial -1)
        highest_file=$(ls "${start_name}"/Occupancy_*.dat 2>/dev/null | grep -v "Occupancy_-1.dat" | sort -V | tail -1)
        if [ -n "$highest_file" ]; then
            echo "Using initial file: $highest_file"
            ./lattice2D-Lea-setinit $density $tumble_rate $total_time $run_name "$highest_file"
        else
            echo "Warning: No final occupancy files found in $start_name, simulation may fail"
            ./lattice2D-Lea-setinit $density $tumble_rate $total_time $run_name "${start_name}/Occupancy_1.dat"
        fi
        
        if [ $? -ne 0 ]; then
            echo "Warning: Simulation failed for $run_name"
        else
            echo "Completed: $run_name"
        fi
    done
done

echo "Parameter sweep completed in directory: $RUNS_DIR"
echo "Run 'python visualize_simulation.py $RUNS_DIR' to create visualizations"
