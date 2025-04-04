#!/usr/bin/env python3
"""
Qubix Time Dilation Simulation — Proof of Concept (with Logging)
----------------------------------------------------------------
This script illustrates a toy model of "quantum collapses" in a 3D voxel grid,
with optional gravitational and velocity-based time dilation.
It saves frames, GIFs, 3D memory scatter plots, and logs parameters to console & file.
Also creates a combined histogram of memory across modes, saved as PNG and GIF.
"""

import os
import csv
import time
import random
import logging
import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Needed for 3D scatter in matplotlib
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SHOW_PLOTS = True           # Whether to show plots interactively
SAVE_FRAMES_AS_GIF = True   # Whether to save frames and compile GIF
LOG_FILE_PATH = "outputs/simulation_log.txt"

# ------------------------------------------------------------------------------
# 3D PLOTTING FUNCTION
# ------------------------------------------------------------------------------
def visualize_3d_memory(grid, parameter='memory', show=True, save_path=None):
    """
    Create a 3D scatter plot of the grid, colored by a chosen parameter (e.g. memory).
    Uses matplotlib's 3D projection.
    """
    xs, ys, zs, colors = [], [], [], []
    for (x, y, z), qubix in grid.grid.items():
        xs.append(x)
        ys.append(y)
        zs.append(z)
        val = getattr(qubix, parameter, 0.0)
        colors.append(val)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='plasma', s=4, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.5, label=parameter.title())
    ax.set_title(f"3D Grid colored by {parameter}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# TIME-DILATION & COLLAPSE HELPERS
# ------------------------------------------------------------------------------
def gravitational_factor(pos: Tuple[int, int, int],
                        center: Tuple[float, float, float],
                        G: float,
                        M: float,
                        c: float) -> float:
    """
    Simplified gravitational time dilation factor:
        sqrt(1 - 2GM/(rc^2))
    """
    r = np.linalg.norm(np.array(pos) - np.array(center))
    r = max(r, 1e-5)
    return np.sqrt(max(1 - (2 * G * M) / (r * c**2), 0.0))

def velocity_factor(velocity: np.ndarray, c: float) -> float:
    """
    Special-relativistic time dilation factor:
        sqrt(1 - v^2 / c^2)
    """
    v = np.linalg.norm(velocity)
    v = min(v, 0.9999 * c)
    return np.sqrt(1 - v**2 / c**2)

def combined_dilation(pos: Tuple[int,int,int],
                      velocity: np.ndarray,
                      config: "SimulationConfig") -> float:
    """
    Combine gravitational and velocity time-dilation factors, depending on active mode.
    """
    factor = 1.0
    if ("gravity" in config.active_modes) or ("combined" in config.active_modes):
        factor *= gravitational_factor(pos, config.gravity_center, config.G,
                                      config.M, config.speed_of_light)
    if ("velocity" in config.active_modes) or ("combined" in config.active_modes):
        factor *= velocity_factor(velocity, config.speed_of_light)
    return factor

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
class SimulationConfig:
    """
    Holds parameters controlling the entire simulation.
    """
    def __init__(self,
                 grid_size=(50,50,50),
                 max_steps=100,
                 measure_prob=0.02,
                 high_speed_fraction=0.6,
                 high_speed_value=0.9):
        self.speed_of_light = 1.0
        self.G = 1e-3
        self.M = 1e3

        # Probability that a measurement/collapse happens in each step, per voxel
        self.measure_prob = measure_prob

        # Number of steps
        self.max_steps = max_steps

        # 3D volume size
        self.grid_size = grid_size

        # Gravity center
        self.gravity_center = tuple(s / 2 for s in self.grid_size)

        # Which modes are active
        self.active_modes: List[str] = []

        # Velocity
        self.high_speed_fraction = high_speed_fraction
        self.high_speed_value = high_speed_value

# ------------------------------------------------------------------------------
# QUBIX & GRID
# ------------------------------------------------------------------------------
class Qubix:
    """
    A single 'quantum voxel' in the grid.
    """

    def __init__(self, pos: Tuple[int,int,int], config: SimulationConfig):
        self.pos = pos
        self.config = config
        self.memory = 0.0
        self.collapse_count = 0
        self.collapsed_at: Optional[float] = None
        self.quantum_time = 0.0

        # Adjust measure_prob if gravity or combined
        if ("gravity" in config.active_modes) or ("combined" in config.active_modes):
            dist = np.linalg.norm(np.array(pos) - np.array(config.gravity_center))
            self.measure_prob = config.measure_prob * np.exp(-0.03 * dist)
        else:
            self.measure_prob = config.measure_prob

    def try_collapse(self, local_time: float) -> bool:
        """
        Attempt a collapse. If random < measure_prob, we 'collapse' and
        increment collapse_count, set memory randomly, etc.
        """
        if random.random() < self.measure_prob:
            self.collapse_count += 1
            self.collapsed_at = local_time
            self.quantum_time += local_time
            # random memory in [0.7, 1.0]
            self.memory = 0.7 + 0.3 * random.random()
            return True
        else:
            # Memory decays if no collapse
            self.memory = max(0.0, self.memory - 0.001)
            return False

class QubixGrid:
    """
    Manages the 3D array of Qubix objects, velocities, and data maps.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid: Dict[Tuple[int,int,int], Qubix] = {}
        self.velocities: Dict[Tuple[int,int,int], np.ndarray] = {}

        self.collapse_density_map = np.zeros(config.grid_size)
        self.local_time_map = np.zeros(config.grid_size)
        self.memory_map = np.zeros(config.grid_size)

        for x in range(config.grid_size[0]):
            for y in range(config.grid_size[1]):
                for z in range(config.grid_size[2]):
                    pos = (x,y,z)
                    self.grid[pos] = Qubix(pos, config)

                    if ("velocity" in config.active_modes) or ("combined" in config.active_modes):
                        if random.random() < config.high_speed_fraction:
                            speed = np.random.uniform(0.4, config.high_speed_value)
                            direction = np.random.normal(0,1,3)
                            direction /= (np.linalg.norm(direction) + 1e-8)
                            self.velocities[pos] = direction * speed
                        else:
                            self.velocities[pos] = np.zeros(3)
                    else:
                        self.velocities[pos] = np.zeros(3)

    def step(self, step_number: int) -> List[float]:
        local_times = []
        for pos, qubix in self.grid.items():
            velocity = self.velocities[pos]
            dilation = combined_dilation(pos, velocity, self.config)
            local_t = step_number * dilation
            if qubix.try_collapse(local_t):
                self.collapse_density_map[pos] += 1
            self.local_time_map[pos] = local_t
            local_times.append(local_t)
        return local_times

    def finalize_maps(self):
        for pos, qubix in self.grid.items():
            self.memory_map[pos] = qubix.memory

    def get_mean_memory(self) -> float:
        return np.mean([q.memory for q in self.grid.values()])

    def get_quantum_time(self) -> List[float]:
        return [q.quantum_time for q in self.grid.values()]

    def export_csv(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "collapse_count", "last_collapsed_at"])
            for (x,y,z), qubix in self.grid.items():
                writer.writerow([x, y, z, qubix.collapse_count, qubix.collapsed_at])

    def save_frame(self, step_num: int, out_dir: str):
        """
        Saves a single 2D slice (Z-mid) of 'memory' as a frame for potential GIF creation.
        """
        os.makedirs(out_dir, exist_ok=True)
        z_mid = self.config.grid_size[2] // 2
        slice_data = np.zeros((self.config.grid_size[0], self.config.grid_size[1]))
        for (x,y,z), qubix in self.grid.items():
            if z == z_mid:
                slice_data[x,y] = qubix.memory
        plt.imshow(slice_data.T, origin='lower', cmap='plasma', vmin=0, vmax=1)
        plt.title(f"Memory z={z_mid}, step={step_num}")
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"frame_{step_num:03d}.png"))
        plt.close()

    def save_heatmaps(self, out_dir: str, show=False):
        """
        Save 2D heatmaps for collapse density and local time at the Z-mid slice.
        """
        os.makedirs(out_dir, exist_ok=True)
        z_mid = self.config.grid_size[2] // 2

        # collapse density
        plt.figure()
        slice1 = self.collapse_density_map[:,:,z_mid].T
        plt.imshow(slice1, origin='lower', cmap='plasma')
        plt.colorbar(label="Collapse Count")
        plt.title(f"Collapse Density (z={z_mid})")
        plt.savefig(os.path.join(out_dir, "collapse_density.png"))
        if show:
            plt.show()
        plt.close()

        # local time
        plt.figure()
        slice2 = self.local_time_map[:,:,z_mid].T
        plt.imshow(slice2, origin='lower', cmap='viridis')
        plt.colorbar(label="Local Time")
        plt.title(f"Local Time (z={z_mid})")
        plt.savefig(os.path.join(out_dir, "time_dilation_heatmap.png"))
        if show:
            plt.show()
        plt.close()


# ------------------------------------------------------------------------------
# SIMULATION FLOW
# ------------------------------------------------------------------------------
def run_simulation(modes: List[str]):
    """
    Run simulation for given modes, returning (steps, local_times, quantum_times, QubixGrid).
    """
    config = SimulationConfig()  # You can customize e.g. config=SimulationConfig(grid_size=(30,30,30), max_steps=150) etc.
    config.active_modes = modes

    grid = QubixGrid(config)
    steps, local_avgs, quantum_avgs = [], [], []

    logger.info("Running mode: %s", ", ".join(modes).upper())
    start = time.time()
    for step_num in range(1, config.max_steps+1):
        local_times = grid.step(step_num)
        steps.append(step_num)
        local_avgs.append(np.mean(local_times))
        quantum_avgs.append(np.mean(grid.get_quantum_time()))

        # If saving frames for a GIF
        if SAVE_FRAMES_AS_GIF:
            frame_dir = os.path.join("outputs", "_".join(modes), "frames")
            grid.save_frame(step_num, frame_dir)

    grid.finalize_maps()
    logger.info("Mode %s done in %.2f sec", modes, time.time()-start)
    return steps, local_avgs, quantum_avgs, grid


def create_gif_from_frames(frame_dir: str, output_path: str):
    frames = []
    for fname in sorted(os.listdir(frame_dir)):
        if fname.endswith(".png"):
            frames.append(imageio.imread(os.path.join(frame_dir, fname)))
    imageio.mimsave(output_path, frames, fps=5)
    logger.info("Saved GIF: %s", output_path)

# ------------------------------------------------------------------------------
# ADDITIONAL: Overlaid Memory Histogram
# ------------------------------------------------------------------------------
def plot_combined_memory_histogram(results):
    """
    Overlaid histogram of final 'memory' values across all modes, shown inline.
    Also saves as a PNG, then converts to single-frame GIF.
    """
    plt.figure(figsize=(8, 5))
    
    for mode, (_, _, _, grid) in results.items():
        # Gather memory from the entire grid for this mode
        memories = [voxel.memory for voxel in grid.grid.values()]
        plt.hist(memories, bins=30, alpha=0.5, label=mode)
    
    plt.xlabel("Memory")
    plt.ylabel("Count")
    plt.title("Final Memory Distribution Across Modes")
    plt.legend()
    plt.grid(True)

    # Save as PNG
    histogram_png = "outputs/final_memory_histogram.png"
    plt.savefig(histogram_png)

    # Show the plot interactively if desired
    plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # For reproducibility
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)

    # Example: run all four modes in a single run
    all_modes = ["standard", "gravity", "velocity", "combined"]
    results = {}

    # Prepare output/log dir
    os.makedirs("outputs", exist_ok=True)
    
    with open("outputs/simulation_log.txt", "a") as log_file:
        logger.info("=== Qubix Time Dilation Simulation (POC) ===")
        start_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("Start Time: %s", start_time_str)
        
        # Write header info to log
        log_file.write("=== Qubix Time Dilation Simulation (POC) ===\n")
        log_file.write(f"Start Time: {start_time_str}\n")
        log_file.write(f"Random Seed: {seed_val}\n")

        # Log default config parameters
        dummy_config = SimulationConfig()
        log_file.write(f"Grid Size: {dummy_config.grid_size}\n")
        log_file.write(f"Max Steps: {dummy_config.max_steps}\n")
        log_file.write(f"G: {dummy_config.G}, M: {dummy_config.M}, c: {dummy_config.speed_of_light}\n")
        log_file.write(f"Measure Prob (Base): {dummy_config.measure_prob}\n")
        log_file.write("--------------------------------------\n")

    for mode in all_modes:
        steps, local_avgs, quantum_avgs, grid = run_simulation([mode])
        results[mode] = (steps, local_avgs, quantum_avgs, grid)

        out_dir = os.path.join("outputs", mode)
        os.makedirs(out_dir, exist_ok=True)

        # Export final CSV
        grid.export_csv(os.path.join(out_dir, "qubix_collapse.csv"))

        # Heatmaps for Z-mid slice
        grid.save_heatmaps(out_dir=out_dir, show=SHOW_PLOTS)

        # GIF creation from frames
        if SAVE_FRAMES_AS_GIF:
            frame_dir = os.path.join(out_dir, "frames")
            gif_path = os.path.join(out_dir, "collapse_memory.gif")
            create_gif_from_frames(frame_dir, gif_path)

        # 3D scatter of memory
        if SHOW_PLOTS:
            logger.info("Creating 3D memory scatter for %s", mode.upper())
            visualize_3d_memory(
                grid,
                parameter='memory',
                show=True,
                save_path=os.path.join(out_dir, "3d_memory.png")
            )

        # Print & log final stats
        mean_mem = grid.get_mean_memory()
        q_times = grid.get_quantum_time()
        delta = np.mean(q_times) - grid.config.max_steps
        logger.info("Final Memory Stats for %s: avg=%.3f", mode.upper(), mean_mem)
        logger.info("Avg Δ (Quantum - Global) for %s: %.3f", mode.upper(), delta)

        with open("outputs/simulation_log.txt", "a") as log_file:
            log_file.write(f"[{mode.upper()}] Memory Avg: {mean_mem:.3f},  Δ(Quantum - Global): {delta:.3f}\n")

    # Compare local times across modes in a single plot
    plt.figure()
    for mode, (steps, local_avgs, _, _) in results.items():
        plt.plot(steps, local_avgs, label=f"{mode} (avg local)")
    plt.plot(steps, steps, '--', color='black', label="Global")
    plt.xlabel("Global Steps")
    plt.ylabel("Local Time (avg)")
    plt.title("Local Time vs. Global, All Modes")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/time_comparison_all_modes.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    # Overlaid memory histogram (also saved as PNG + single-frame GIF)
    plot_combined_memory_histogram(results)

    logger.info("=== Simulation Complete ===")
    with open("outputs/simulation_log.txt", "a") as log_file:
        log_file.write("=== Simulation Complete ===\n\n")


if __name__ == "__main__":
    main()
