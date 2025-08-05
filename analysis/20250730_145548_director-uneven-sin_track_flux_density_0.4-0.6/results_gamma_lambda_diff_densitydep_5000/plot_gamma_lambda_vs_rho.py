import numpy as np
import matplotlib.pyplot as plt

# Load data from file, skipping header
filename = "analysis/20250730_145548_director-uneven-sin_track_flux_density/results_gamma_lambda_diff_densitydep_5000/gamma_lambda_results_diff_densitydep.txt"
data = []
with open(filename, "r") as f:
    for line in f:
        if line.strip() and not line.startswith("slice"):
            parts = line.split()
            if len(parts) == 4:
                # slice, rho, gamma, lambda
                data.append([float(parts[1]), float(parts[2]), float(parts[3])])

data = np.array(data)
rho = data[:, 0]
gamma = data[:, 1]
lambda_ = data[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(rho, gamma, marker='o', label='gamma')
plt.plot(rho, lambda_, marker='s', label='lambda')
plt.xlabel('rho')
plt.ylabel('Value')
plt.title('Gamma and Lambda vs. Rho')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(filename.replace('.txt', '_plot.png'))
plt.show()
