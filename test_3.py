import os
import re
import numpy as np

run_dir_avg = "/Users/leabauer/Documents/code/runs/run_20250808_100929_director-uneven-sin_track_flux_density_densavg5000_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5"  # set to the correct folder
run_dir_normal = "/Users/leabauer/Documents/code/runs/run_20250808_101408_director-uneven-sin_track_flux_density_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5"  # set to the correct folder
density_avg_path = os.path.join(run_dir_avg, "Density_avg_start5000.dat")

DensityAveragingStart = 5000
SaveInterval = 9

# Find all density files
pattern = re.compile(r"Density_(\d+)\.dat")
steps = []
file_paths = []

for fname in os.listdir(run_dir_normal):
    match = pattern.match(fname)
    if match:
        step = int(match.group(1))
        if step >= DensityAveragingStart and step % SaveInterval == 0:
            steps.append(step)
            file_paths.append(os.path.join(run_dir_normal, fname))

# Sort
sorted_pairs = sorted(zip(steps, file_paths))
sorted_steps, sorted_paths = zip(*sorted_pairs)

print(f"Using {len(sorted_steps)} steps from {sorted_steps[0]} to {sorted_steps[-1]}")

# Compute mean
profiles = [np.loadtxt(fp) for fp in sorted_paths]
avg = np.mean(profiles, axis=0)
sim_avg = np.loadtxt(density_avg_path)

diff = np.abs(avg - sim_avg)
print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))

