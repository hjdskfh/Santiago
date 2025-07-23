import numpy as np
import matplotlib.pyplot as plt

# Constants
Lx = 200
Gamma = -0.5
G = 1.0
X_max = 0
lower_bound = 0.0
upper_bound = 1.0

# Golden section search
def golden_section_search(func, a, b, tol=1e-6):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if func(c) < func(d):
            a = c
        else:
            b = d
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

# Potential functions
def uneven_sin(x): return np.sin(x + X_max) + Gamma * np.sin(2 * (x + X_max))
def shifted_uneven_sin(x): return G * uneven_sin(x) + 0.5
def symmetric_sin(x): return G * np.sin(x + X_max) + 0.5

# Rescaling
def rescaling_function(func, x, lower_bound, upper_bound, f_max, f_min):
    # This function is no longer needed for vectorized rescaling
    pass

# Initialize and return move probability
def initialize_move_prob_map(func, label):
    x_vals = np.linspace(0, 2*np.pi, 1000)
    global X_max
    X_max = golden_section_search(func, 0.0, 2 * np.pi)
    pot_arr = func(x_vals)
    f_max = np.max(pot_arr)
    f_min = np.min(pot_arr)
    if abs(f_max - f_min) < 1e-12:
        move_prob = np.full_like(pot_arr, 0.5)
    else:
        move_prob = lower_bound + (pot_arr - f_min) / (f_max - f_min) * (upper_bound - lower_bound)
    print(f"x_vals: {x_vals[:5]}...")  # Print first 5 values for debugging
    print(f"move_prob: {move_prob[:5]}...")  # Print first 5 values for debugging
    x_box = np.linspace(0, Lx, 1000)
    return x_box, move_prob, label

# Compute and plot
x1, mp1, lbl1 = initialize_move_prob_map(uneven_sin, "Uneven Sin")
x2, mp2, lbl2 = initialize_move_prob_map(shifted_uneven_sin, "Shifted Uneven Sin")
x3, mp3, lbl3 = initialize_move_prob_map(symmetric_sin, "Symmetric Sin")

plt.figure(figsize=(10, 6))
plt.plot(x1, mp1, label=lbl1)
plt.plot(x2, mp2, label=lbl2)
plt.plot(x3, mp3, label=lbl3)
plt.title("Move Probability Maps for Different Potentials")
plt.xlabel("x (from 0 to 2π)")
plt.ylabel("Move Probability")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("move_probability_maps.png")
