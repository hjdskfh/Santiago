import numpy as np
import matplotlib.pyplot as plt

def uneven_sin_function(x, gamma):
    """The uneven sin function used in both potentials"""
    return np.sin(x) + gamma * np.sin(2 * x)

def golden_section_search_numpy(func, a, b, tol=1e-6, gamma=-0.5):
    """Find maximum of function using golden section search"""
    gr = (np.sqrt(5) + 1) / 2  # Golden ratio
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    
    while abs(b - a) > tol:
        if func(c, gamma) > func(d, gamma):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    
    return (a + b) / 2

# Parameters
Lx = 100
gamma = -0.5
G = 0.3

# Find maximum for normalization
x_max = golden_section_search_numpy(uneven_sin_function, 0.0, np.pi, gamma=gamma)
f_max = uneven_sin_function(x_max, gamma)

print(f"Function maximum occurs at x = {x_max:.6f}")
print(f"Maximum value f_max = {f_max:.6f}")

# Create x array for plotting
x_positions = np.arange(Lx)
scale_x = (x_positions / Lx) * 2 * np.pi

# Calculate the underlying function
f_values = uneven_sin_function(scale_x, gamma)

# Calculate movement probabilities for both potential types
uneven_sin_prob = 1 - ((f_values / (2 * f_max)) + 0.5)
director_based_prob = 0.5 + G * (f_values / (2 * f_max))

# Ensure valid probability ranges
uneven_sin_prob = np.clip(uneven_sin_prob, 0.0, 1.0)
director_based_prob = np.clip(director_based_prob, 0.0, 1.0)

# Create the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: The underlying function
ax1.plot(x_positions, f_values, 'b-', linewidth=2, label=f'sin(x) + {gamma}*sin(2x)')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('X Position')
ax1.set_ylabel('Function Value')
ax1.set_title('Underlying Function: sin(x) + γ*sin(2x)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Movement probabilities comparison
ax2.plot(x_positions, uneven_sin_prob, 'r-', linewidth=2, label='Uneven-sin (inverted)')
ax2.plot(x_positions, director_based_prob, 'g-', linewidth=2, label='Director-based-sin (direct)')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Movement Probability')
ax2.set_title('Movement Probability Comparison')
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Uneven-sin as 2D heatmap
uneven_2d = np.tile(uneven_sin_prob, (40, 1))
im3 = ax3.imshow(uneven_2d, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)
ax3.set_title('Uneven-sin Movement Probability (2D)')
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
plt.colorbar(im3, ax=ax3, label='Movement Probability')

# Plot 4: Director-based-sin as 2D heatmap
director_2d = np.tile(director_based_prob, (40, 1))
im4 = ax4.imshow(director_2d, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)
ax4.set_title('Director-based-sin Movement Probability (2D)')
ax4.set_xlabel('X Position')
ax4.set_ylabel('Y Position')
plt.colorbar(im4, ax=ax4, label='Movement Probability')

plt.tight_layout()
plt.savefig('analysis/potential_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some analysis
print(f"\nAnalysis:")
print(f"Gamma = {gamma}, G = {G}")
print(f"Uneven-sin: High function values -> LOW movement probability (clusters form where function is low)")
print(f"Director-based: High function values -> HIGH movement probability (clusters form where function is high)")
print(f"\nTo make clusters form on the same side, we need to invert the director-based formula.")
