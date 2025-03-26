import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from natsort import natsorted

# ----- Qubix class with energy -----
class Qubix:
    def __init__(self, id, position, d=2):
        self.id = id
        self.position = position
        self.d = d
        # Initialize equal amplitude state vector
        norm = 1.0 / np.sqrt(d)
        self.state = [norm + 0j for _ in range(d)]
        self.collapsed_at = None
        # Memory indicates convergence; energy tracks 'dynamical' behavior.
        self.memory = 0.0
        # Energy is an abstract measure; start with a fixed value.
        self.energy = 1.0

    def evolve(self, step, measure_prob=0.1):
        """
        Evolve the state by applying a small phase shift.
        Also, update energy (e.g., slight decay over time).
        With a probability (measure_prob), collapse the state.
        """
        phase = 0.05  # fixed phase shift
        self.state = [amp * cmath.exp(1j * phase) for amp in self.state]
        
        # Update energy: for example, decay a little each step.
        self.energy *= 0.99
        
        # Random chance to collapse (simulate partial measurement)
        if self.collapsed_at is None and random.random() < measure_prob:
            self.collapse(step)
            return True
        return False

    def collapse(self, step):
        """
        Collapse the state probabilistically according to the squared amplitudes.
        Also, release energy or set energy to a lower value upon collapse.
        """
        probs = [abs(amp)**2 for amp in self.state]
        total = sum(probs)
        norm_probs = [p / total for p in probs] if total > 0 else [1.0, 0.0]
        outcome = self.weighted_choice(norm_probs)
        # Collapse: state becomes 1 for the chosen outcome, 0 for the other.
        self.state = [1.0 if i == outcome else 0.0 for i in range(self.d)]
        self.collapsed_at = step
        self.memory = 1.0
        # When a voxel collapses, we could consider that energy is 'released'
        # or simply reset to a baseline lower energy.
        self.energy = 0.2

    def weighted_choice(self, weights):
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r < cumulative:
                return i
        return len(weights) - 1

# ----- QubixGrid class with energy visualization -----
class QubixGrid:
    def __init__(self, size=(5,5,5)):
        self.size = size
        self.grid = {}
        id_counter = 0
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    self.grid[(x, y, z)] = Qubix(id=id_counter, position=(x, y, z))
                    id_counter += 1

    def step(self, step):
        collapsed_this_step = 0
        for voxel in self.grid.values():
            if voxel.collapsed_at is None:
                if voxel.evolve(step):
                    collapsed_this_step += 1
        return collapsed_this_step

    def summary(self):
        total = len(self.grid)
        collapsed = sum(1 for v in self.grid.values() if v.collapsed_at is not None)
        avg_energy = np.mean([v.energy for v in self.grid.values()])
        return total, collapsed, avg_energy

    def build_memory_image(self):
        """
        Build a 2D image for the z=2 slice, showing memory.
        """
        z_level = 2
        mem_array = np.zeros((self.size[1], self.size[0]))
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                voxel = self.grid.get((x, y, z_level))
                if voxel:
                    mem_array[y, x] = voxel.memory
        return mem_array

    def build_energy_image(self):
        """
        Build a 2D image for the z=2 slice, showing energy levels.
        """
        z_level = 2
        energy_array = np.zeros((self.size[1], self.size[0]))
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                voxel = self.grid.get((x, y, z_level))
                if voxel:
                    energy_array[y, x] = voxel.energy
        return energy_array

    def update_noncollapsed_memory(self):
        """
        Increase memory gradually for voxels that haven't collapsed.
        """
        for voxel in self.grid.values():
            if voxel.collapsed_at is None:
                voxel.memory = min(0.5, voxel.memory + 0.02)

# ----- GIF Creation Function -----
def make_gif(frame_dir, output_gif, fps=5):
    frames = []
    file_list = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
    file_list = natsorted(file_list)  # ensure frame order by step

    for file_name in file_list:
        file_path = os.path.join(frame_dir, file_name)
        frames.append(imageio.imread(file_path))

    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"Saved GIF: {output_gif}")

# ----- Main simulation with energy and visualization -----
def main():
    steps = 100
    grid = QubixGrid(size=(5,5,5))
    output_dir_mem = "simulation_memory_frames"
    output_dir_energy = "simulation_energy_frames"
    os.makedirs(output_dir_mem, exist_ok=True)
    os.makedirs(output_dir_energy, exist_ok=True)
    
    print("Starting simulation with energy tracking and visualization...")
    for step in range(steps):
        collapsed = grid.step(step)
        grid.update_noncollapsed_memory()
        total, overall_collapsed, avg_energy = grid.summary()
        print(f"Step {step+1}: Collapsed this step = {collapsed}, Total collapsed = {overall_collapsed}/{total}, Avg Energy = {avg_energy:.3f}")
        
        # Memory visualization for z=2
        mem_image = grid.build_memory_image()
        plt.imshow(mem_image, cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.title(f"Memory (z=2) at Step {step+1}")
        plt.colorbar(label="Memory")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir_mem, f"step_{step+1:03d}.png"))
        plt.close()
        
        # Energy visualization for z=2
        energy_image = grid.build_energy_image()
        plt.imshow(energy_image, cmap='hot', origin='lower', vmin=0, vmax=1)
        plt.title(f"Energy (z=2) at Step {step+1}")
        plt.colorbar(label="Energy")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir_energy, f"step_{step+1:03d}.png"))
        plt.close()

        if overall_collapsed == total:
            print("All voxels have collapsed. Ending simulation early.")
            break

    # After simulation, create GIFs from the saved frames.
    make_gif(output_dir_mem, "memory_animation.gif", fps=5)
    make_gif(output_dir_energy, "energy_animation.gif", fps=5)

if __name__ == "__main__":
    main()
