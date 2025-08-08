import numpy as np
import glob

files = sorted(glob.glob("/Users/leabauer/Documents/code/runs/run_20250808_101408_director-uneven-sin_track_flux_density_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5/Density_*.dat"))
relevant_files = [f for f in files if 5000 <= int(f.split("_")[-1].split(".")[0]) <= 11000]

profiles = [np.loadtxt(f) for f in relevant_files]
avg = np.mean(profiles, axis=0)

sim_avg = np.loadtxt("/Users/leabauer/Documents/code/runs/run_20250808_100929_director-uneven-sin_track_flux_density_densavg5000_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5/Density_avg_start5000.dat")

print("Max diff:", np.max(np.abs(avg - sim_avg)))