import numpy as np
import glob

import os

run_dir_avg = "/Users/leabauer/Documents/code/runs/run_20250808_100929_director-uneven-sin_track_flux_density_densavg5000_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5"  # set to the correct folder
run_dir_normal = "/Users/leabauer/Documents/code/runs/run_20250808_101408_director-uneven-sin_track_flux_density_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5"  # set to the correct folder
start = 5004
end = 11000
interval = 9

profiles = []
missing = []

for step in range(start, end + 1, interval):
    fname = os.path.join(run_dir_normal, f"Density_{step}.dat")
    if os.path.exists(fname):
        data = np.loadtxt(fname)
        profiles.append(data)
    else:
        missing.append(step)

if missing:
    print("Warning: missing density files for steps:", missing)

avg = np.mean(profiles, axis=0)

sim_avg = np.loadtxt(os.path.join(run_dir_avg, "Density_avg_start5000.dat"))

diff = np.abs(avg - sim_avg)
print("Max diff:", np.max(diff))
print("Mean diff:", np.mean(diff))
