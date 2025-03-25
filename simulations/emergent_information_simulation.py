#!/usr/bin/env python3
"""
Emergent Informational Universe:
Hybrid Classical Simulation with Repulsive Vector Field
Memory Map

This simulation demonstrates a toy model for an emergent informational universe.
Local collapse events and memory feedback reconfigure a probability (or energy)
field, while a vector field—derived from local gradients—is updated with:
  1. Repulsion: Each vector is pushed away from the local average, encouraging divergence.
  2. Saturation ("Pressure Release"): If vector magnitudes exceed a threshold, they are scaled down,
     preventing runaway buildup.

The simulation uses a real-valued grid (0–1 values) with the original 3×3 collapse mechanism.
It displays four subplots:
  - The grid (flipped horizontally for alignment)
  - A quiver plot of the vector field (colored by memory)
  - A memory (scaffold) heatmap (no gradient arrows)
  - A metrics plot (entropy and total energy)

Scaling is set to macroscopic (energy_scale = 1e3).
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
import matplotlib.cm as cm

# ------------------------
# PARAMETERS & SETTINGS
# ------------------------
grid_size = 50
steps = 200
noise_std = 0.05
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Collapse & energy update parameters (original style)
base_collapse_prob = 0.04       # Base collapse probability
memory_collapse_scale = 0.06    # Additional probability from memory
energy_injection = 10           # Energy added during a collapse event
energy_transfer_rate = 0.02     # Diffusion rate for energy updates
dissipation_rate = 0.005        # Nonlinear dissipation factor

# Memory & feedback parameters
memory_feedback_factor = 0.3
memory_accumulation = 0.5
memory_decay_rate = 0.98
memory_threshold = 0.8         # Threshold for amplifying memory influence

# Hybrid collapse weighting
hybrid_weight = 0.5

# Vector field parameters
max_vector_speed = 0.1         # Maximum allowed vector speed
vector_decay = 0.95            # Decay factor for previous vector field contributions
vector_smoothing_kernel = torch.tensor([
    [0.05, 0.1, 0.05],
    [0.1,  0.4, 0.1],
    [0.05, 0.1, 0.05]
], dtype=torch.float32)

# Additional simulation parameters
spread_intensity = 0.1
reversal_interval = 30
reversal_strength = 0.05

# Local vector shift scale (for energy warping)
local_shift_scale = 0.05

# Visualization scale for quiver (exaggerates the small raw vectors)
quiver_scale_factor = 100

# Vector field saturation: if magnitude exceeds this, scale it down ("pressure release")
vector_field_saturation = 0.08

# Macroscopic scaling for energy
energy_scale = 1e3

# Collapse Mode Settings:
# Set collapse_mode to "simple" or "standard"
collapse_mode = "standard"  
# For simple mode: collapse every collapse_interval frames
collapse_interval = 10 
# For standard mode: define the range and number of collapses
collapse_range = (20, 180)  # Frames between which collapses can occur
collapse_count = 10         # Total collapses to occur in that range

# ------------------------
# SETUP: DIRECTORIES & DEVICE
# ------------------------
os.makedirs("data", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# INITIALIZATION
# ------------------------
grid_shape = (1, 1, grid_size, grid_size)
grid = torch.rand(grid_shape, device=device)  # values between 0 and 1
energy = torch.zeros(grid_shape, device=device)  # start with zero energy
memory_mask = torch.zeros(grid_shape, dtype=torch.bool, device=device)
spread_mask = torch.zeros(grid_shape, device=device)
reversal_mask = torch.zeros(grid_shape, device=device)
vector_field = torch.zeros((1, 2, grid_size, grid_size), device=device)
memory_feedback = torch.zeros(grid_shape, device=device)

# CSV logs for entropy and total energy
entropy_csv = "improved_entropy_energy.csv"
collapse_csv = "improved_collapse_details.csv"
with open(entropy_csv, "w", newline="") as f:
    csv.writer(f).writerow(["Step", "Entropy", "Total Energy"])
with open(collapse_csv, "w", newline="") as f:
    csv.writer(f).writerow(["Step", "Collapse_X", "Collapse_Y"])

entropy_values = []
total_energy_values = []
collapse_data = []  # List to log collapse events (frame, x, y)

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def calculate_entropy(grid_tensor):
    """Compute Shannon entropy of the grid values."""
    flat = grid_tensor.view(-1)
    flat = torch.clamp(flat, 1e-10, 1.0)
    return -torch.sum(flat * torch.log2(flat)).item()

def calculate_total_energy(energy_tensor):
    """Return the sum of the energy field."""
    return energy_tensor.sum().item()

def update_vector_field(energy, vector_field):
    """
    Compute local gradients of the energy field, smooth them, and blend with the existing vector field.
    Also, apply a saturation to release excess pressure.
    """
    grad_y = energy[:, :, 2:, 1:-1] - energy[:, :, :-2, 1:-1]
    grad_x = energy[:, :, 1:-1, 2:] - energy[:, :, 1:-1, :-2]
    grad_x = F.pad(grad_x, (1, 1, 1, 1))
    grad_y = F.pad(grad_y, (1, 1, 1, 1))
    new_vector = torch.cat((grad_x, grad_y), dim=1)
    magnitude = torch.norm(new_vector, dim=1, keepdim=True)
    factor = torch.where(magnitude > max_vector_speed, max_vector_speed / magnitude, torch.ones_like(magnitude))
    new_vector = new_vector * factor

    kernel = vector_smoothing_kernel.view(1, 1, 3, 3).to(device)
    new_vx = F.conv2d(new_vector[:, 0:1], kernel, padding=1)
    new_vy = F.conv2d(new_vector[:, 1:2], kernel, padding=1)
    new_vector = torch.cat((new_vx, new_vy), dim=1)
    vector_field = vector_field * vector_decay + new_vector * (1 - vector_decay)

    # Saturation: if any vector exceeds a threshold, scale it down
    current_mags = torch.norm(vector_field, dim=1, keepdim=True)
    saturation_factor = torch.clamp(vector_field_saturation / current_mags, max=1.0)
    vector_field = vector_field * saturation_factor

    with torch.no_grad():
        vf_magnitude = torch.norm(vector_field, dim=1)
        print(f"  [Debug] Vector Field mean={vf_magnitude.mean():.6f}, max={vf_magnitude.max():.6f}")
    return vector_field

def local_vector_shift(energy, vector_field, scale=0.05):
    """
    Shift each cell's energy based on its local vector direction via grid_sample.
    """
    N, C, H, W = energy.shape
    assert N == 1, "Batch size must be 1."
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )
    base_grid = torch.stack((x_coords, y_coords), dim=-1).unsqueeze(0)  # (1, H, W, 2)
    vx = vector_field[:, 0] * scale * 2 / W
    vy = vector_field[:, 1] * scale * 2 / H
    offset = torch.stack((vx, vy), dim=-1)
    new_grid = base_grid + offset
    shifted_energy = F.grid_sample(energy, new_grid, align_corners=False)
    return shifted_energy

def spread_energy(energy, vector_field=None):
    """
    Update the energy field via diffusion and local vector shift.
    """
    diffusion_rate = energy_transfer_rate
    kernel = torch.tensor([[0.0, 1/4, 0.0],
                           [1/4, 0.0, 1/4],
                           [0.0, 1/4, 0.0]], dtype=torch.float32, device=device).view(1,1,3,3)
    diffusion = F.conv2d(energy, kernel, padding=1)
    new_energy = energy - energy * diffusion_rate + diffusion * diffusion_rate

    if vector_field is not None:
        shifted = local_vector_shift(new_energy, vector_field, scale=local_shift_scale)
        new_energy = new_energy * 0.9 + shifted * 0.1

    new_energy = new_energy * (1 - dissipation_rate)
    return torch.clamp(new_energy, 0, 10)

def update_memory_feedback(memory_feedback, collapse_event_mask):
    """
    Update memory feedback by decaying previous memory and adding new collapse events.
    """
    return memory_feedback * memory_decay_rate + collapse_event_mask * memory_accumulation

def update_grid(grid, energy, memory_mask, spread_mask, reversal_mask, vector_field, memory_feedback, frame):
    """
    Update the grid by averaging local cells, adding fluctuations, and applying memory feedback.
    Then update the vector field and energy field accordingly.
    """
    new_grid = grid.clone()
    new_energy = energy.clone()
    collapse_event_mask = torch.zeros_like(grid, device=device)

    for i in range(grid_size):
        for j in range(grid_size):
            cell_mem = memory_feedback[0, 0, i, j]
            if frame % reversal_interval == 0 and reversal_mask[0, 0, i, j].item() != 0:
                new_grid[0, 0, i, j] = new_grid[0, 0, i, j] * (1 - reversal_strength) + reversal_mask[0, 0, i, j] * reversal_strength
            elif spread_mask[0, 0, i, j].item() > 0:
                new_grid[0, 0, i, j] = new_grid[0, 0, i, j] * (1 - spread_intensity) + spread_mask[0, 0, i, j] * spread_intensity
                spread_mask[0, 0, i, j] = spread_mask[0, 0, i, j] * 0.9
            else:
                i0, i1 = max(0, i-1), min(grid_size, i+2)
                j0, j1 = max(0, j-1), min(grid_size, j+2)
                local_avg = grid[0, 0, i0:i1, j0:j1].mean()
                fluctuation = torch.normal(0.0, noise_std, (1,), device=device)
                feedback_multiplier = 2.0 if cell_mem > memory_threshold else 1.0
                feedback_term = cell_mem * memory_feedback_factor * feedback_multiplier
                new_val = local_avg + fluctuation + feedback_term
                new_grid[0, 0, i, j] = torch.clamp(new_val, 0, 1)

    vector_field = update_vector_field(energy, vector_field)
    new_energy = spread_energy(energy, vector_field=vector_field)
    memory_feedback = update_memory_feedback(memory_feedback, collapse_event_mask)

    return new_grid, new_energy, memory_mask, spread_mask, reversal_mask, vector_field, memory_feedback

def collapse_grid(grid, energy, frame, memory_mask, spread_mask, reversal_mask, memory_feedback):
    """
    Perform a collapse event in a random 3×3 region, injecting energy and updating memory.
    This function preserves the original collapse behavior.
    """
    new_grid = grid.clone()
    new_energy = energy.clone()
    collapse_x = np.random.randint(0, grid_size - 3)
    collapse_y = np.random.randint(0, grid_size - 3)
    print(f"Quantum Collapse at Step {frame}!")
    print(f"  Region: x=[{collapse_x}:{collapse_x+3}], y=[{collapse_y}:{collapse_y+3}]")
    reversal_mask[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = new_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3]
    discrete_grid = new_grid.clone()
    discrete_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = (
        discrete_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] >= 0.5
    ).float()
    new_energy[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] += energy_injection
    continuous_region = new_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3]
    blended = (1 - hybrid_weight) * continuous_region + hybrid_weight * discrete_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3]
    new_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = blended
    new_energy = spread_energy(new_energy, vector_field=None)
    collapse_prob_dynamic = base_collapse_prob + (frame / steps) * 0.02
    collapse_prob_dynamic = min(collapse_prob_dynamic, 0.2)
    region_vals = new_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3]
    mask = torch.rand(region_vals.shape, device=device) < collapse_prob_dynamic
    region_vals[mask] = (region_vals[mask] >= 0.5).float()
    new_grid[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = region_vals
    memory_mask[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = True
    spread_mask[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = 1.0
    collapse_event_mask = torch.zeros_like(grid, device=device)
    collapse_event_mask[:, :, collapse_x:collapse_x+3, collapse_y:collapse_y+3] = 1.0
    memory_feedback = update_memory_feedback(memory_feedback, collapse_event_mask)
    collapse_data.append([frame, collapse_x, collapse_y])
    with open(collapse_csv, "a", newline="") as f:
        csv.writer(f).writerow([frame, collapse_x, collapse_y])
    return new_grid, new_energy, memory_mask, spread_mask, reversal_mask, memory_feedback

def get_arrow_colors(memory_feedback, colormap_name="coolwarm"):
    """
    Normalize the memory feedback field and return a flattened RGBA array for each arrow.
    """
    mem = memory_feedback.cpu().numpy().squeeze()  # shape: (grid_size, grid_size)
    norm = (mem - mem.min()) / (mem.max() - mem.min() + 1e-8)
    cmap = plt.get_cmap(colormap_name)
    colors = cmap(norm)  # shape: (grid_size, grid_size, 4)
    return colors.reshape(-1, 4)

# ------------------------
# FIGURE & PLOTTING SETUP
# ------------------------
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])
ax_sim = fig.add_subplot(gs[0, 0])
ax_quiver = fig.add_subplot(gs[0, 1])
ax_memory = fig.add_subplot(gs[0, 2])
ax_metric = fig.add_subplot(gs[1, :])
plt.tight_layout()

# ------------------------
# ANIMATE FUNCTION (No Memory Gradient Quiver)
# ------------------------
def animate(frame):
    global grid, energy, memory_mask, spread_mask, reversal_mask, vector_field, memory_feedback
    print(f"Processing frame {frame} / {steps}")
    
    # 1) Update the grid and energy based on collapses, memory feedback, etc.
    grid, energy, memory_mask, spread_mask, reversal_mask, vector_field, memory_feedback = update_grid(
        grid, energy, memory_mask, spread_mask, reversal_mask, vector_field, memory_feedback, frame
    )
    
    # 2) Collapse logic based on collapse_mode (simple, standard, or default random)
    if collapse_mode == "simple":
        if frame % collapse_interval == 0 and frame > 0:
            grid, energy, memory_mask, spread_mask, reversal_mask, memory_feedback = collapse_grid(
                grid, energy, frame, memory_mask, spread_mask, reversal_mask, memory_feedback
            )
    elif collapse_mode == "standard":
        start, end = collapse_range
        if start <= frame <= end:
            collapse_frames = np.linspace(start, end, collapse_count, dtype=int)
            if frame in collapse_frames:
                grid, energy, memory_mask, spread_mask, reversal_mask, memory_feedback = collapse_grid(
                    grid, energy, frame, memory_mask, spread_mask, reversal_mask, memory_feedback
                )
    else:
        if steps - 10 > 30:
            collapse_frames = np.sort(
                np.random.choice(
                    np.arange(30, steps-10), 
                    size=np.random.randint(4, 8),
                    replace=False
                )
            )
        else:
            collapse_frames = []
        if frame in collapse_frames:
            grid, energy, memory_mask, spread_mask, reversal_mask, memory_feedback = collapse_grid(
                grid, energy, frame, memory_mask, spread_mask, reversal_mask, memory_feedback
            )
    
    # 3) Compute and log metrics
    ent = calculate_entropy(grid)
    tot_energy = calculate_total_energy(energy)
    entropy_values.append(ent)
    total_energy_values.append(tot_energy)
    with open(entropy_csv, "a", newline="") as f:
        csv.writer(f).writerow([frame, ent, tot_energy])
    
    # 4) Save snapshots periodically
    if frame % 10 == 0:
        np.save(os.path.join("data", f"snapshot_grid_frame_{frame}.npy"), grid.cpu().numpy())
        np.save(os.path.join("data", f"snapshot_energy_frame_{frame}.npy"), energy.cpu().numpy())
        print(f"Saved snapshots for frame {frame}")
    
    # Clear the subplot axes
    ax_sim.clear()
    ax_quiver.clear()
    ax_memory.clear()
    ax_metric.clear()
    
    # 5) Grid (flip horizontally for alignment)
    ax_sim.imshow(grid.cpu().squeeze(), cmap="viridis", interpolation="nearest", origin='lower')
    ax_sim.set_title(f"Quantum Grid (Step {frame})")
    
    # 6) Quiver: Overlay vector field with arrow colors mapped to memory feedback
    vector_field_np = vector_field.cpu().numpy().squeeze()
    vx = vector_field_np[0]
    vy = vector_field_np[1]
    X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    arrow_colors = get_arrow_colors(memory_feedback)  # memory-based color
    ax_quiver.quiver(
        X, Y,
        vx * quiver_scale_factor,
        vy * quiver_scale_factor,
        color=arrow_colors,
        scale_units='xy',
        scale=1,
        width=0.005,
        headwidth=3
    )
    ax_quiver.set_title("Vector Field (Scaled, colored by Memory)")
    ax_quiver.set_xlim(0, grid_size)
    ax_quiver.set_ylim(0, grid_size)
    
    # 7) Memory Feedback: Show the memory field (scaffold) (No gradient quiver overlay)
    mem = memory_feedback.cpu().numpy().squeeze()
    ax_memory.imshow(mem, cmap="plasma", interpolation="nearest", origin='lower')
    ax_memory.set_title("Memory Feedback (Scaffold)")
    
    # 8) Metrics Plot: Entropy & total energy over time
    ax1 = ax_metric
    ax2 = ax_metric.twinx()
    ln1 = ax1.plot(entropy_values, label="Entropy", color="blue")
    ln2 = ax2.plot(total_energy_values, label="Energy", color="red")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Entropy", color="blue")
    ax2.set_ylabel("Energy", color="red")
    
    if total_energy_values and max(total_energy_values) > 0:
        ax2.set_ylim(0, max(total_energy_values) * 1.1)
    else:
        ax2.set_ylim(0, 1)
    
    lns = ln1 + ln2
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc="upper right")
    ax1.set_title(f"Metrics (Step {frame}): Ent={ent:.1f}, Energy={tot_energy:.2e}")
    
    plt.tight_layout()
    return []

anim = FuncAnimation(fig, animate, frames=steps, interval=200, blit=False)
anim.save("emergent_information_simulation.gif", writer="pillow")
print("Animation saved as 'emergent_information_simulation.gif'")
plt.show()


# ------------------------
# POST-SIMULATION ANALYSIS (PCA, etc.)
# ------------------------
print("Loading entropy/energy logs...")
entropy_data = np.genfromtxt(entropy_csv, delimiter=",", skip_header=1)
print("First 5 rows of logs:")
print(entropy_data[:5])
print("Loading grid snapshots...")
grid_snapshots = []
for frame in range(0, steps, 10):
    path = os.path.join("data", f"snapshot_grid_frame_{frame}.npy")
    if os.path.exists(path):
        snap = np.load(path)
        grid_snapshots.append(snap.flatten())
        print(f"Snapshot {frame} shape:", snap.shape)
print(f"Loaded {len(grid_snapshots)} grid snapshots.")
print("Loading energy snapshots...")
energy_snapshots = []
for frame in range(0, steps, 10):
    path = os.path.join("data", f"snapshot_energy_frame_{frame}.npy")
    if os.path.exists(path):
        snap = np.load(path)
        energy_snapshots.append(snap.flatten())
        print(f"Energy snapshot {frame} shape:", snap.shape)
print(f"Loaded {len(energy_snapshots)} energy snapshots.")
if len(grid_snapshots) > 1:
    print("Performing PCA on grid snapshots...")
    grid_snapshots_arr = np.vstack(grid_snapshots)
    pca = PCA(n_components=2)
    grid_pca = pca.fit_transform(grid_snapshots_arr)
    print("Grid PCA variance explained:", pca.explained_variance_ratio_)
    print("Performing PCA on energy snapshots...")
    energy_snapshots_arr = np.vstack(energy_snapshots)
    pca_energy = PCA(n_components=2)
    energy_pca = pca_energy.fit_transform(energy_snapshots_arr)
    print("Energy PCA variance explained:", pca_energy.explained_variance_ratio_)
    print("Clustering grid snapshots using KMeans...")
    kmeans = KMeans(n_clusters=2, random_state=random_seed)
    clusters = kmeans.fit_predict(grid_snapshots_arr)
    print("Grid clusters:", clusters)
    num_snapshots = grid_pca.shape[0]
    entropy_snapshots_log = np.array(entropy_values)[::10][:num_snapshots]
    total_energy_snapshots_log = np.array(total_energy_values)[::10][:num_snapshots]
    pc1 = grid_pca[:, 0]
    corr_entropy, pval_entropy = pearsonr(pc1, entropy_snapshots_log)
    corr_energy, pval_energy = pearsonr(pc1, total_energy_snapshots_log)
    print(f"Correlation PC1 vs. Entropy: {corr_entropy:.3f} (p={pval_entropy:.3g})")
    print(f"Correlation PC1 vs. Energy: {corr_energy:.3f} (p={pval_energy:.3g})")
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sc1 = plt.scatter(grid_pca[:,0], grid_pca[:,1], c=total_energy_snapshots_log, cmap="viridis")
    plt.colorbar(sc1, label="Energy")
    plt.title("Grid PCA colored by Energy")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.subplot(1,2,2)
    sc2 = plt.scatter(grid_pca[:,0], grid_pca[:,1], c=entropy_snapshots_log, cmap="magma")
    plt.colorbar(sc2, label="Entropy")
    plt.title("Grid PCA colored by Entropy")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()
    np.save("grid_pca.npy", grid_pca)
    np.save("energy_pca.npy", energy_pca)
    np.savetxt("grid_clusters.csv", clusters, delimiter=",", header="Cluster", comments="")
    with open("pca_correlation.txt", "w") as f:
        f.write(f"Correlation PC1 vs. Entropy: {corr_entropy:.3f} (p={pval_entropy:.3g})\n")
        f.write(f"Correlation PC1 vs. Energy: {corr_energy:.3f} (p={pval_energy:.3g})\n")
print("Done with simulation and analysis!")
