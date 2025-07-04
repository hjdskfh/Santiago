import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

folder = 'Test2'
filename = 'Occupancy_00001.dat'

filepath = f'./{folder}/{filename}'  # build the path using the variable

data = np.loadtxt(filepath)
print(f"Data shape: {data.shape}")
print(f"Data range: {data.min()} to {data.max()}")

# Create a colormap visualization
plt.figure(figsize=(12, 6))

# Create a dynamic colormap based on actual data range
max_occupancy = int(data.max())
min_occupancy = int(data.min())

# Generate colors automatically using a colormap
cmap = plt.cm.viridis  # You can change this to other colormaps like 'plasma', 'inferno', 'Blues', etc.

# Plot the occupancy matrix
im = plt.imshow(data.T, cmap=cmap, origin='lower', aspect='equal', 
                vmin=min_occupancy, vmax=max_occupancy)
plt.colorbar(im, label='Occupancy Level')
plt.title(f'Lattice Occupancy - {filename}')
plt.xlabel('X Position')
plt.ylabel('Y Position')

# Add grid for better visualization
plt.grid(True, alpha=0.3, linewidth=0.5)

# Save the plot instead of showing it
output_filename = f'{folder}/occupancy_plot_{filename[:-4]}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_filename}")

# Show the plot (optional - comment out if you only want to save)
plt.tight_layout()
plt.show()