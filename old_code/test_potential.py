import numpy as np
import matplotlib.pyplot as plt

def uneven_sin_function(x, gamma):
    return np.sin(x) + gamma * np.sin(2 * x)

# Test with gamma = -0.5
gamma = -0.5
x = np.linspace(0, 2*np.pi, 1000)
y = uneven_sin_function(x, gamma)

# Find maximum
x_max_idx = np.argmax(y)
x_max = x[x_max_idx]
f_max = y[x_max_idx]

print(f"Function: sin(x) + {gamma} * sin(2*x)")
print(f"Maximum occurs at x = {x_max:.4f}, f_max = {f_max:.4f}")
print(f"Function range: [{y.min():.4f}, {y.max():.4f}]")

# Calculate the movement probability as in the C code
# MoveProbMap[x][y] = 1 - ((uneven_sin_function(scale_x, Gamma) / (2 * f_max)) + 0.5)
move_prob = 1 - ((y / (2 * f_max)) + 0.5)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', linewidth=2, label=f'sin(x) + {gamma}*sin(2x)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=f_max, color='r', linestyle='--', alpha=0.5, label=f'Max = {f_max:.3f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Uneven Sin Function')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(x, move_prob, 'r-', linewidth=2, label='Movement Probability')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Movement Probability')
plt.title('Movement Probability = 1 - (f(x)/(2*f_max) + 0.5)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

# Add vertical lines to show lattice positions for a 40x40 grid
lattice_positions = np.linspace(0, 2*np.pi, 40, endpoint=False)
for pos in lattice_positions[::4]:  # Show every 4th position for clarity
    plt.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/leabauer/Documents/code/potential_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Also show what the 2D pattern should look like
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Create a 40x40 grid like in the simulation
Lx, Ly = 40, 40
x_grid = np.arange(Lx)
y_grid = np.arange(Ly)

# Calculate movement probability for each x position
move_prob_2d = np.zeros((Ly, Lx))  # Note: first dimension is y, second is x for numpy arrays
for i in range(Lx):
    scale_x = (i / Lx) * 2 * np.pi
    prob = 1 - ((uneven_sin_function(scale_x, gamma) / (2 * f_max)) + 0.5)
    # Ensure valid probability range
    prob = max(0.0, min(1.0, prob))
    move_prob_2d[:, i] = prob  # Fill entire column with same probability

# Plot the 2D pattern
im1 = ax1.imshow(move_prob_2d, cmap='viridis', origin='lower', aspect='equal')
ax1.set_title('2D Movement Probability Pattern\n(as stored in file)')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
plt.colorbar(im1, ax=ax1)

# Plot the 1D profile
ax2.plot(x_grid, move_prob_2d[0, :], 'b-', linewidth=2)
ax2.set_xlabel('X Position')
ax2.set_ylabel('Movement Probability')
ax2.set_title('1D Profile across X direction')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/leabauer/Documents/code/2d_potential_pattern.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nMovement probabilities at key positions:")
for i in [0, 10, 20, 30, 39]:
    scale_x = (i / Lx) * 2 * np.pi
    prob = 1 - ((uneven_sin_function(scale_x, gamma) / (2 * f_max)) + 0.5)
    prob = max(0.0, min(1.0, prob))
    print(f"x={i}: scale_x={scale_x:.3f}, prob={prob:.3f}")
