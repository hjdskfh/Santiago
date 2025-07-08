import numpy as np
from scipy.optimize import minimize_scalar

# Define the function
def f(x):
    return np.sin(x) - 0.5 * np.sin(2 * x)

# Use negative f(x) to find the maximum via minimization
result = minimize_scalar(lambda x: -f(x), bounds=(0, np.pi), method='bounded')

# Extract max value and corresponding x
x_max = result.x
f_max = f(x_max)

print(f"Maximum value: {f_max}")
print(f"At x = {x_max}")
