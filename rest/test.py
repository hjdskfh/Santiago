import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

# Generate random discrete data (0, 1, or 2)
np.random.seed(42)
data = np.random.choice([0, 1, 2], size=(10, 10))

# Example discrete colormap with 3 levels: 0 (white), 1 (yellow), 2 (red)
cmap = ListedColormap(['white', 'yellow', 'red'])
norm = plt.Normalize(vmin=0, vmax=2)

im = plt.imshow(data, cmap=cmap, norm=norm)
cbar = plt.colorbar(im, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['0', '1', '2'])

# Add a yellow dot and '1' to the legend
legend_elements = [Patch(facecolor='yellow', edgecolor='black', label='1')]
plt.legend(handles=legend_elements, loc='upper right', title='Value')

plt.show()