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
echo "Compiling lattice2D-Lea-sweep..."
gcc -o lattice2D-Lea-sweep lattice2D-Lea-sweep.c -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting parameter sweep..."

# Parameter ranges - modify these as needed
densities=(0.1 0.3 0.5 0.7 0.9 1.1 1.3)
tumble_rates=(0.001 0.005 0.01 0.05 0.1 0.2)
total_time=10000

# Counter for progress
total_runs=$((${#densities[@]} * ${#tumble_rates[@]}))
current_run=0

# Loop through all parameter combinations
for density in "${densities[@]}"; do
    for tumble_rate in "${tumble_rates[@]}"; do
        current_run=$((current_run + 1))
        
        # Create run name and put it in the runs directory
        run_name="$RUNS_DIR/d${density}_t${tumble_rate}_time${total_time}"
        
        echo "[$current_run/$total_runs] Running: density=$density, tumble_rate=$tumble_rate, run_name=$run_name"
        
        # Run the simulation
        ./lattice2D-Lea-sweep $density $tumble_rate $total_time $run_name
        
        if [ $? -ne 0 ]; then
            echo "Warning: Simulation failed for $run_name"
        else
            echo "Completed: $run_name"
        fi
    done
done

echo "Parameter sweep completed in directory: $RUNS_DIR"
echo "Run 'python visualize_simulation.py $RUNS_DIR' to create visualizations"
