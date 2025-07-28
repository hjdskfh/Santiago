import numpy as np

def print_nonzero_values(dat_file):
    arr = np.loadtxt(dat_file)
    nonzero_indices = np.nonzero(arr)[0]
    if len(nonzero_indices) == 0:
        print("No nonzero values found.")
    else:
        print(f"Nonzero values in {dat_file}:")
        for idx in nonzero_indices:
            print(f"Index: {idx}, Value: {arr[idx]}")

# Example usage:
print_nonzero_values("/Users/leabauer/Documents/code/runs/run_20250725_152641_uneven-sin_track_flux_density_one/d0.000125_t0.07_time500000_gamma-0.5_g1/Density_14.dat")