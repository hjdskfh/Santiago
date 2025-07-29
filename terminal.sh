#!/bin/bash

# Exit immediately if a command fails
set -e

# Compile the C code
echo "Compiling simulation..."
gcc -Wall -Wextra -pedantic -O3 -o lattice2D-Lea-4-potential lattice2D-Lea-4-potential.c -lm

# Make the secondary script executable
chmod +x /Users/leabauer/Documents/code/run_4_potential.sh

echo "Starting parameter sweep..."
echo "Available options: --track-movement, --track-flux, --track-density, --move-prob TYPE"
echo ""

# Assumption for Potential: only input lower and upperbound symmetric around 0.5, maximum at 0
# Run the initialization script with the new flags
# Modify the line below to customize your run #uneven-sin, director-uneven-sin, director-symmetric-sin #track-movement, track-density, track-flux
time ./run_4_potential.sh --move-prob uneven-sin --track-movement --track-flux --track-density "$@" 

