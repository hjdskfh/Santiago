#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting parameter sweep..."
echo "Available options: --track-movement, --track-flux, --track-density, --move-prob TYPE (uneven-sin, director-uneven-sin, director-symmetric-sin), --density-averaging STEP"
echo ""

# Run the initialization script with the new flags
# Modify the line below to customize your run 
# possible to average the density already during the simulation with --density-averaging

# Loop over amplitudes
time {

for amplitude in 0.05 0.075; do # 0.10 0.125 0.15 0.175 0.20
  echo "Running with amplitude $amplitude..."
  time ./run.sh --move-prob director-uneven-sin --track-movement --track-flux --track-density --density-averaging 5000 --amplitude "$amplitude" "$@"
done

}


