import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/Users/leabauer/Documents/code/runs/run_20250723_164719_uneven-sin_track_flux_density/d0.00125_t0.2_time100000_gamma-0.5_g1/XAccumulatedFlux_100000.dat")
plt.plot(data, marker='o', linestyle='-')
plt.xlabel("X Position")
plt.ylabel("Accumulated Flux")
plt.title("XAccumulatedFlux at Final Step")
plt.grid(True, alpha=0.3)
plt.show()