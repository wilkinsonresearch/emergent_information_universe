#!/usr/bin/env python3
"""
Display Concepts from the Emergent Informational Universe

This script demonstrates three simplified visualizations to illustrate key concepts:
1. Evolved Probability Field with Successive Collapses.
2. Particle Traversal in the Quantum Field.
3. A 2D Vector Landscape with Random Collapses that Alter Arrow Directions.

These images serve as blog illustrations of emergent structures, potential field evolution,
and how local collapse events can redirect flows in an informational universe.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ------------------------
# GLOBAL SETTINGS
# ------------------------
NUM_PARTICLES = 10  # <--- Change this to set how many particles are used in particle traversal

# ---------------------------------------------------
# 1. Evolved Probability Field with Successive Collapses
# ---------------------------------------------------
def evolve_probability_field(size=100, num_steps=6):
    # Initialize base probability field as a 2D Gaussian
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    sigma = 2.0
    base_probability = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    base_probability /= base_probability.max()  # normalize

    probability_field = base_probability.copy()
    collapse_history = []
    for step in range(num_steps):
        # Choose a collapse point near the center with a decreasing offset range
        offset_range = max(1, 6 - step)
        collapse_point = (
            size // 2 + np.random.randint(-offset_range, offset_range),
            size // 2 + np.random.randint(-offset_range, offset_range)
        )
        collapse_history.append(collapse_point)
        
        # Create a flow field from the collapse point
        flow = np.zeros_like(probability_field)
        for i in range(size):
            for j in range(size):
                dx = i - collapse_point[0]
                dy = j - collapse_point[1]
                distance = np.sqrt(dx**2 + dy**2) + 1e-6
                flow[i, j] = np.cos(distance / 2) / distance
        
        # Apply the flow with diminishing effect
        probability_field += (0.3 / (step + 1)) * flow
        probability_field = np.clip(probability_field, 0, 1)
    
    return X, Y, probability_field, collapse_history

def plot_probability_field():
    X, Y, field, collapse_history = evolve_probability_field()
    plt.figure(figsize=(8, 6))
    plt.title("Evolved Probability Field with Successive Collapses")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.pcolormesh(X, Y, field, cmap='plasma', shading='auto')
    plt.colorbar(label="Probability")
    
    # Plot collapse history as white circles
    for (cy, cx) in collapse_history:
        # Map index to coordinate in [-5,5]
        plt.plot(np.linspace(-5, 5, field.shape[1])[cx],
                 np.linspace(-5, 5, field.shape[0])[cy],
                 'wo', markersize=5)
    
    plt.grid(True, linestyle=':')
    plt.savefig("evolved_probability_field.png", dpi=150)
    plt.show()

# ---------------------------------------------------
# 2. Particle Traversal in the Quantum Field
# ---------------------------------------------------
def compute_quantum_field(particles, field, size):
    influence_map = np.zeros((size, size))
    for py, px in particles:
        for i in range(size):
            for j in range(size):
                dx = i - py
                dy = j - px
                distance = np.sqrt(dx**2 + dy**2) + 1e-6
                influence_map[i, j] += np.cos(distance / 2) / distance
    return field + 0.5 * influence_map

def particle_traversal(field, size=100, num_particles=5, steps=10):
    """
    Evolve 'field' by moving 'num_particles' over 'steps' iterations.
    'num_particles' is set by the global variable NUM_PARTICLES by default.
    """
    particle_positions = np.random.randint(10, size-10, size=(num_particles, 2))
    trajectories = [particle_positions.copy()]
    
    for _ in range(steps):
        quantum_field = compute_quantum_field(particle_positions, field, size)
        for idx, (py, px) in enumerate(particle_positions):
            if 1 < py < size-2 and 1 < px < size-2:
                gx = (quantum_field[py, px+1] - quantum_field[py, px-1]) / 2
                gy = (quantum_field[py+1, px] - quantum_field[py-1, px]) / 2
                # Move particle in the direction of increasing field (gradient ascent)
                new_px = np.clip(int(px + gx * 1.5), 1, size-2)
                new_py = np.clip(int(py + gy * 1.5), 1, size-2)
                particle_positions[idx] = [new_py, new_px]
        trajectories.append(particle_positions.copy())
    return quantum_field, trajectories

def plot_particle_traversal():
    _, _, field, _ = evolve_probability_field(size=100, num_steps=6)
    # Use the global NUM_PARTICLES variable here
    quantum_field, trajectories = particle_traversal(field, size=100, num_particles=NUM_PARTICLES, steps=10)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    plt.figure(figsize=(8, 6))
    plt.title("Quantum Field with Moving Particles")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.imshow(quantum_field, cmap='plasma', extent=[-5, 5, -5, 5], origin='lower')
    plt.colorbar(label="Quantum Potential")
    
    # Plot particle trajectories
    trajectories = np.array(trajectories)
    for particle in trajectories.transpose(1, 0, 2):  # each particle's trajectory
        xs = [x[pos[1]] for pos in particle]
        ys = [y[pos[0]] for pos in particle]
        plt.plot(xs, ys, marker='o', markersize=4, linewidth=1.2)
    
    plt.grid(True)
    plt.savefig("particle_traversal.png", dpi=150)
    plt.show()

# ---------------------------------------------------
# 3. 2D Vector Landscape with Random Collapses that Change Arrow Directions
# ---------------------------------------------------
def vector_landscape(size=30):
    # Create a coordinate domain from -5 to 5
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Define an initial vector field (simple swirl or harmonic field)
    U = np.cos(Y)
    V = np.sin(X)
    
    # Introduce random collapse events that modify the vector field
    num_collapses = 5
    collapse_coords = np.random.randint(5, size-5, size=(num_collapses, 2))
    for cx, cy in collapse_coords:
        radius = 3
        # For each collapse, reassign vectors in the region to point radially outward from the collapse center
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                ix, iy = cx + dx, cy + dy
                if 0 <= ix < size and 0 <= iy < size:
                    # Compute vector pointing from collapse center to cell
                    diff_x = ix - cx
                    diff_y = iy - cy
                    norm = np.sqrt(diff_x**2 + diff_y**2) + 1e-6
                    # Set new arrow values; you can choose a scaling factor
                    U[iy, ix] = (diff_x / norm) * np.abs(U[iy, ix])
                    V[iy, ix] = (diff_y / norm) * np.abs(V[iy, ix])
    return X, Y, U, V, collapse_coords

def plot_vector_landscape():
    X, Y, U, V, collapse_coords = vector_landscape(size=30)
    plt.figure(figsize=(8, 8))
    plt.title("2D Vector Landscape with Random Collapses")
    plt.quiver(X, Y, U, V, color='mediumseagreen')
    # Mark collapse points with red dots
    for cx, cy in collapse_coords:
        plt.plot(X[0, cx], Y[cy, 0], 'ro')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig("vector_landscape.png", dpi=150)
    plt.show()

# ---------------------------------------------------
# Main Dispatch
# ---------------------------------------------------
if __name__ == "__main__":
    print("Generating Evolved Probability Field image...")
    plot_probability_field()
    
    print(f"Generating Particle Traversal image with {NUM_PARTICLES} particles...")
    plot_particle_traversal()
    
    print("Generating Vector Landscape image...")
    plot_vector_landscape()
    
    print("Images created!")
