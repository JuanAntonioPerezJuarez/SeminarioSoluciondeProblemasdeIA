import numpy as np
import matplotlib.pyplot as plt
from math import gamma, sin, pi  # Importar desde math directamente

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_steps = 1000
alpha = 1.5  # Lévy index (1 < alpha <= 2, smaller = heavier tail)

# Generate Lévy flight steps
def levy_flight(n, alpha):
    """Generate Lévy flight steps using the Mantegna method"""
    # Generate random variables from normal distribution
    sigma_u = (gamma(1 + alpha) * sin(pi * alpha / 2) / 
               (gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
    
    u = np.random.normal(0, sigma_u, n)
    v = np.random.normal(0, 1, n)
    
    steps = u / (np.abs(v)**(1 / alpha))
    return steps

# Generate 2D Lévy flight
x_steps = levy_flight(n_steps, alpha)
y_steps = levy_flight(n_steps, alpha)

# Calculate cumulative positions
x = np.cumsum(x_steps)
y = np.cumsum(y_steps)

# Create the plot
plt.figure(figsize=(12, 5))

# Plot 1: The flight path
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=0.5, alpha=0.6)
plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
plt.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
plt.title(f'Lévy Flight Path (α={alpha})', fontsize=14, fontweight='bold')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Plot 2: Step size distribution
plt.subplot(1, 2, 2)
step_sizes = np.sqrt(x_steps**2 + y_steps**2)
plt.hist(step_sizes, bins=50, edgecolor='black', alpha=0.7)
plt.title('Step Size Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Step Size')
plt.ylabel('Frequency')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('levy_flight.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Lévy Flight generated with {n_steps} steps")
print(f"Total distance traveled: {np.sum(step_sizes):.2f}")
print(f"Displacement from start: {np.sqrt(x[-1]**2 + y[-1]**2):.2f}")
