import numpy as np
import matplotlib.pyplot as plt

def print_nonzero_values(dat_file):
    arr = np.loadtxt(dat_file)
    plt.plot(arr)
    plt.title(f"Data from {dat_file}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.savefig("moveprobmap_plot.png")
    nonzero_indices = np.nonzero(arr)[0]
    if len(nonzero_indices) == 0:
        print("No nonzero values found.")
    else:
        print(f"Nonzero values in {dat_file}:")
        for idx in nonzero_indices:
            print(f"Index: {idx}, Value: {arr[idx]}")

# Example usage:
print_nonzero_values("/Users/leabauer/Documents/code/runs/run_20250729_181115_director-uneven-sin_track_flux_density/d0.7_t0.07_time100000_gamma-0.5/MoveProbgradU_1.dat")