#!/bin/bash

# Exit immediately if a command fails
set -e

# Compile the C code
gcc -O3 -o lattice2D-Lea-4-potential lattice2D-Lea-4-potential.c -lm

# Make the secondary script executable
chmod +x /Users/leabauer/Documents/code/run_4_potential.sh

# Run the initialization script with correct flag
# Note: --move-prob requires a value (default, uneven-sin)
./run_4_potential.sh --move-prob uneven-sin 
#--start-config --move-prob uneven-sin 
