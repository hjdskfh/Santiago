import matplotlib.pyplot as plt

# Replace 'data.dat' with your actual filename
filename = '/Users/leabauer/Documents/code/runs/run_20250808_104054_director-uneven-sin_track_flux_density_densavg5000_amplitude0.20/d0.7_t0.11_time11000_gamma-0.5/Density_avg_start5000.dat'

# Read the data from the file
with open(filename, 'r') as f:
    line = f.readline()
    values = list(map(float, line.split()))

# Plot the values
plt.plot(values, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot from data.dat')
plt.grid(True)
plt.show()
