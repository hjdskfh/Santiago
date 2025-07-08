#!/bin/bash

# Exit immediately if a command fails
set -e

# Compile the C code
gcc -O3 -o lattice2D-Lea-4-potential lattice2D-Lea-4-potential.c -lm

# Make the secondary script executable
chmod +x /Users/leabauer/Documents/code/run_4_potential.sh

# Run the initialization script with correct flags
# Note: --move-prob requires a value (default, uneven-sin, director-based-sin)
# Use --save-interval and --track-movement for movement analysis
./run_4_potential.sh --move-prob director-based-sin --track-movement --calculate-density-and-flux 1500
#./run_4_potential.sh --move-prob uneven-sin --save-interval 1000 --track-movement
#./run_4_potential.sh --start-config --move-prob uneven-sin --save-interval 1000 --track-movement 
