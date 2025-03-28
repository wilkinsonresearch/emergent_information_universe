#!/usr/bin/env python3
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

# ------------------------
# 1. Simulation Configuration
# ------------------------
class SimulationConfig:
    def __init__(self):
        self.speed_of_light = 1.0  # c=1 in simulation units
        self.memory_alpha = 0.5
        self.collapse_beta = 0.3
        self.base_measure_prob = 0.10
        self.memory_decay_rate = 0.99
        self.memory_increase_factor = 0.5
        
        # Simplified GR parameters (illustrative)
        self.G = 1e-3
        self.M = 1e3

# ------------------------
# 2. Qubix (Voxel) Class
# ------------------------
class Qubix:
    def __init__(self, q_id, position, role='excite', config=None):
        self.id = q_id
        self.position = position
        self.role = role
        self.config = config if config is not None else SimulationConfig()
        self.measure_prob = self.config.base_measure_prob
        self.memory = 0.0
        self.collapsed_at = []
        self.quantum_time = 0.0

    def attempt_collapse(self, local_time):
        if random.random() < self.measure_prob:
            self.collapsed_at.append(local_time)
            self.quantum_time += local_time
            return True
        return False

    def update_memory(self, did_collapse, local_reward=0.0):
        if did_collapse:
            self.memory += self.config.memory_increase_factor * (1.0 + local_reward)
        self.memory *= self.config.memory_decay_rate

# ------------------------
# 3. BufferQubix Class
# ------------------------
class BufferQubix:
    """
    Global buffer that adjusts its probabilities based on avg quantum time,
    producing a correction factor to mitigate runaway time dilation.
    """
    def __init__(self, alpha=0.1, collapse_threshold=0.8):
        self.probabilities = {"1": 0.5, "2": 0.5}
        self.alpha = alpha
        self.collapse_threshold = collapse_threshold
        self.collapsed = False
        self.state = None
        self.quantum_time = 0.0

    def update(self, avg_quantum_time):
        if avg_quantum_time > 1.0:
            self.probabilities["1"] += self.alpha * (1 - self.probabilities["1"])
            self.probabilities["2"] -= self.alpha * self.probabilities["2"]
        else:
            self.probabilities["1"] -= self.alpha * self.probabilities["1"]
            self.probabilities["2"] += self.alpha * (1 - self.probabilities["2"])

        # small random noise
        self.probabilities["1"] += random.uniform(-0.005, 0.005)
        self.probabilities["2"] += random.uniform(-0.005, 0.005)

        # normalize
        total = self.probabilities["1"] + self.probabilities["2"]
        self.probabilities["1"] /= total
        self.probabilities["2"] /= total

        if self.probabilities["1"] >= self.collapse_threshold:
            self.collapsed = True
            self.state = 1
        elif self.probabilities["2"] >= self.collapse_threshold:
            self.collapsed = True
            self.state = 2

        correction_factor = self.probabilities["1"] - self.probabilities["2"]
        return correction_factor

# ------------------------
# 4. QubixGrid Class
# ------------------------
class QubixGrid:
    def __init__(self, size=(10, 10, 10), config=None, gravity_well=None, multi_gravity_wells=None):
        """
        multi_gravity_wells: list of dicts, each with 'center', 'radius', 'effect', 'mass'
        """
        self.config = config if config is not None else SimulationConfig()
        self.size = size
        self.grid = {}
        q_id = 0
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    role = 'inhibit' if random.random() < 0.3 else 'excite'
                    self.grid[(x, y, z)] = Qubix(q_id, (x, y, z), role=role, config=self.config)
                    q_id += 1

        self.tree = cKDTree(list(self.grid.keys()))
        self.total_qubix_count = len(self.grid)
        self.gravity_well = gravity_well
        self.multi_gravity_wells = multi_gravity_wells
        self.collapse_density = {}  # track collapses per voxel

    def compute_gravitational_dilation_factor(self, q):
        if not self.gravity_well and not self.multi_gravity_wells:
            return 1.0

        def well_factor(center, mass, pos_arr):
            G = self.config.G
            c = self.config.speed_of_light
            r = np.linalg.norm(pos_arr - center)
            r = max(r, 1e-5)
            arg = 1 - (2*G*mass)/(r*c*c)
            return max(arg, 0.0)

        pos_arr = np.array(q.position)
        factors = []
        if self.gravity_well:
            center = np.array(self.gravity_well['center'])
            mass = self.gravity_well['mass']
            factors.append(well_factor(center, mass, pos_arr))
        if self.multi_gravity_wells:
            for w in self.multi_gravity_wells:
                center = np.array(w['center'])
                mass = w['mass']
                factors.append(well_factor(center, mass, pos_arr))

        product = 1.0
        for f in factors:
            product *= np.sqrt(f)
        return product

    def total_time_dilation_factor(self, q, velocity):
        grav_factor = self.compute_gravitational_dilation_factor(q)
        c = self.config.speed_of_light
        v_sq = np.dot(velocity, velocity)
        if v_sq >= c*c:
            v_sq = 0.9999*c*c
        gamma = 1.0 / np.sqrt(1.0 - v_sq/(c*c))
        sr_factor = 1.0 / gamma
        return grav_factor * sr_factor

    def step(self, global_step, velocity_dict=None):
        if velocity_dict is None:
            velocity_dict = {}
        collapsed_flags = {}
        local_time_record = {}

        for pos, q in self.grid.items():
            velocity = velocity_dict.get(pos, np.array([0.0, 0.0, 0.0]))
            dilation_factor = self.total_time_dilation_factor(q, velocity)
            local_time = global_step * dilation_factor

            # Gravity well influence
            if self.gravity_well:
                center = np.array(self.gravity_well['center'])
                radius = self.gravity_well['radius']
                effect = self.gravity_well['effect']
                dist = np.linalg.norm(np.array(pos) - center)
                if dist <= radius:
                    influence = effect*(1 - dist/radius)
                    q.measure_prob = min(1.0, q.measure_prob*(1+influence))
                    q.memory += influence

            # Multi-wells
            if self.multi_gravity_wells:
                for w in self.multi_gravity_wells:
                    center = np.array(w['center'])
                    radius = w['radius']
                    effect = w['effect']
                    dist = np.linalg.norm(np.array(pos) - center)
                    if dist <= radius:
                        influence = effect*(1 - dist/radius)
                        q.measure_prob = min(1.0, q.measure_prob*(1+influence))
                        q.memory += influence

            did_collapse = q.attempt_collapse(local_time)
            collapsed_flags[pos] = did_collapse
            local_time_record[pos] = local_time

            if did_collapse:
                self.collapse_density[pos] = self.collapse_density.get(pos, 0) + 1

        # Update memory
        for pos, q in self.grid.items():
            q.update_memory(collapsed_flags[pos], local_reward=0.0)

        return collapsed_flags, local_time_record

    def move_particles(self, velocity_dict, dt=1.0, fixed_positions=None):
        new_velocity_dict = {}
        updated_grid = {}
        for pos, q in list(self.grid.items()):
            # If position is in fixed_positions, do not move it
            if fixed_positions and pos in fixed_positions:
                updated_grid[pos] = q
                new_velocity_dict[pos] = velocity_dict.get(pos, np.array([0.0,0.0,0.0]))
            else:
                v = velocity_dict.get(pos, np.array([0.0,0.0,0.0]))
                new_pos = (
                    int(round(pos[0] + v[0]*dt)),
                    int(round(pos[1] + v[1]*dt)),
                    int(round(pos[2] + v[2]*dt))
                )
                if 0 <= new_pos[0]<self.size[0] and 0<=new_pos[1]<self.size[1] and 0<=new_pos[2]<self.size[2]:
                    updated_grid[new_pos] = q
                    new_velocity_dict[new_pos] = v
        self.grid = updated_grid
        return new_velocity_dict

    def apply_buffer_correction(self, correction_factor):
        for q in self.grid.values():
            q.quantum_time *= (1 - 0.1*correction_factor)

    # ---------------- Visualization Methods ----------------
    def visualize_gravitational_potential(self):
        if not self.gravity_well and not self.multi_gravity_wells:
            print("No gravity well defined; skipping gravitational potential visualization.")
            return
        if self.multi_gravity_wells:
            w = self.multi_gravity_wells[0]
            center = np.array(w['center'])
            mass = w['mass']
        else:
            center = np.array(self.gravity_well['center'])
            mass = self.gravity_well['mass']
        potential = np.zeros(self.size)
        G = self.config.G
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                for z in range(self.size[2]):
                    pos = np.array([x,y,z])
                    r = np.linalg.norm(pos - center)
                    r = max(r,1e-5)
                    potential[x,y,z] = -G*mass/r
        z_slice = self.size[2] // 2
        plt.figure(figsize=(6,5))
        plt.imshow(potential[:,:,z_slice], cmap="inferno", origin="lower")
        plt.title("Gravitational Potential (z-slice)")
        plt.colorbar(label="Potential")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def visualize_velocity_field(self, velocity_dict, z_slice=9):
        print(f"Velocities at z={z_slice}:")
        for (x,y,z), v in velocity_dict.items():
            if z == z_slice:
                print(f"Pos: {(x,y,z)}, Velocity: {v}")

        X, Y = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)
        for (x,y,z), v in velocity_dict.items():
            if z == z_slice:
                U[y,x] = v[0]
                V[y,x] = v[1]
        plt.figure(figsize=(6,6))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, width=0.01, color='blue')
        plt.title(f"Velocity Field (z={z_slice})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-1, self.size[0]+1)
        plt.ylim(-1, self.size[1]+1)
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.show()

    def visualize_time_dilation_heatmap(self, local_time_record, z_slice=9):
        dilation = np.zeros(self.size)
        for (x,y,z), t in local_time_record.items():
            dilation[x,y,z] = t
        plt.figure(figsize=(6,5))
        plt.imshow(dilation[:,:,z_slice], cmap="viridis", origin="lower")
        plt.title(f"Local Time Dilation (z={z_slice})")
        plt.colorbar(label="Local Time")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def visualize_3d_grid(self, parameter='memory'):
        xs, ys, zs, colors = [],[],[],[]
        for pos, q in self.grid.items():
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            if parameter=='memory':
                colors.append(q.memory)
            elif parameter=='quantum_time':
                colors.append(q.quantum_time)
            else:
                colors.append(0)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=colors, cmap=cm.viridis, s=80)
        plt.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
        ax.set_title(f"3D Voxel Distribution (Colored by {parameter})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def visualize_collapse_density(self):
        if not self.collapse_density:
            print("No collapse data recorded.")
            return
        collapse_map = np.zeros(self.size)
        for pos, count in self.collapse_density.items():
            x,y,z = pos
            collapse_map[x,y,z] = count
        z_slice = self.size[2] // 2
        plt.figure(figsize=(6,5))
        plt.imshow(collapse_map[:,:,z_slice], cmap="plasma", origin="lower")
        plt.title("Collapse Density (z-slice)")
        plt.colorbar(label="Collapse Count")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def advanced_analysis(self):
        pass

# ------------------------
# 5. Steps-Based Test Runner
# ------------------------
class SimulationTestRunner:
    def __init__(self, modes, num_steps=500, grid_size=(10,10,10), high_speed_value=0.9):
        """
        'modes' can include:
          - "standard"
          - "gravity_well"
          - "orbit"
          - "buffer"
          - "high_speed"
        high_speed_value: velocity for high-speed voxels.
        """
        self.modes = modes
        self.num_steps = num_steps
        self.grid_size = grid_size
        self.config = SimulationConfig()
        self.gravity_well = None
        self.acceleration_target = None
        self.velocity_dict = {}
        self.high_speed_value = high_speed_value
        self.multi_gravity_wells = None

        if "gravity_well" in self.modes or "orbit" in self.modes:
            center = (grid_size[0]//2, grid_size[1]//2, grid_size[2]//2)
            self.gravity_well = {'center': center, 'radius': 3, 'effect': 0.5, 'mass': self.config.M}
            print(f"Gravity Well Active: center={center}, radius=3, effect=0.5, mass={self.config.M}")
        if "orbit" in self.modes:
            self.acceleration_target = (center[0], center[1]-4, center[2])
            print(f"Orbit Mode: target voxel at {self.acceleration_target}")
        if "buffer" in self.modes:
            print("Buffer Mode Active: Global buffer will counteract time dilation.")
            self.buffer = BufferQubix(alpha=0.05, collapse_threshold=0.8)
        else:
            self.buffer = None
        if "high_speed" in self.modes:
            print("High Speed Mode Active: We'll assign a high velocity to a voxel away from center.")
        if not self.modes or "standard" in self.modes:
            print("Standard Simulation (plus additional modes).")

        self.global_time_list = []
        self.avg_local_time_list = []
        self.avg_quantum_time_list = []
        self.high_speed_voxel_log = []

    def initialize_velocity_dict(self, grid):
        for pos in grid.grid.keys():
            self.velocity_dict[pos] = np.array([0.0, 0.0, 0.0])
        if "orbit" in self.modes and self.acceleration_target in self.velocity_dict:
            center = np.array(self.gravity_well['center'])
            target = np.array(self.acceleration_target)
            direction = target - center
            perp = np.array([-direction[1], direction[0], 0.0])
            norm = np.linalg.norm(perp)
            perp = perp / norm if norm != 0 else np.array([0.0,0.0,0.0])
            self.velocity_dict[self.acceleration_target] = perp * 0.5

        if "high_speed" in self.modes:
            # Place the high-speed voxel at (2,7,9) so it's away from center
            high_speed_target = (2,7,9)
            self.velocity_dict[high_speed_target] = np.array([self.high_speed_value]*3)
            print(f"High Speed Voxel: {high_speed_target} -> velocity={self.velocity_dict[high_speed_target]}")

    def run(self):
        grid = QubixGrid(size=self.grid_size, config=self.config,
                         gravity_well=self.gravity_well,
                         multi_gravity_wells=self.multi_gravity_wells)

        # We'll fix the high-speed voxel's position so it remains in z=9
        fixed_positions = []
        high_speed_target = (2,7,9) if "high_speed" in self.modes else None
        if high_speed_target:
            fixed_positions.append(high_speed_target)

        self.initialize_velocity_dict(grid)

        for step in range(self.num_steps):
            if "orbit" in self.modes and self.acceleration_target in self.velocity_dict:
                center = np.array(self.gravity_well['center'])
                pos_arr = np.array(self.acceleration_target)
                direction = center - pos_arr
                distance = np.linalg.norm(direction)
                if distance != 0:
                    pull = (direction/distance)*0.05
                    self.velocity_dict[self.acceleration_target] += pull

            collapsed, local_times = grid.step(step, velocity_dict=self.velocity_dict)
            self.global_time_list.append(step)
            avg_local = np.mean(list(local_times.values()))
            self.avg_local_time_list.append(avg_local)
            avg_quantum = np.mean([q.quantum_time for q in grid.grid.values()])
            self.avg_quantum_time_list.append(avg_quantum)

            if self.buffer:
                correction = self.buffer.update(avg_quantum)
                grid.apply_buffer_correction(correction)

            # Keep the high-speed voxel in place
            self.velocity_dict = grid.move_particles(self.velocity_dict, dt=1.0, fixed_positions=fixed_positions)

            if high_speed_target and high_speed_target in grid.grid:
                voxel = grid.grid[high_speed_target]
                self.high_speed_voxel_log.append((step, voxel.memory, voxel.quantum_time))

        diff = np.mean(np.array(self.avg_quantum_time_list) - np.array(self.global_time_list))
        print("Simulation complete.")
        print(f"Final memory distribution: "
              f"Min={min(q.memory for q in grid.grid.values()):.3f}, "
              f"Max={max(q.memory for q in grid.grid.values()):.3f}, "
              f"Avg={np.mean([q.memory for q in grid.grid.values()]):.3f}")
        print(f"Average difference (Quantum Clock - Global Clock)={diff:.3f}")

        if high_speed_target in self.velocity_dict:
            print(f"Final velocity at {high_speed_target}: {self.velocity_dict[high_speed_target]}")
        else:
            print(f"{high_speed_target} not in velocity_dict (possibly left the grid).")

        self.compare_clocks()
        grid.visualize_3d_grid(parameter='memory')
        grid.visualize_gravitational_potential()
        grid.visualize_time_dilation_heatmap(local_times, z_slice=9)
        grid.visualize_velocity_field(self.velocity_dict, z_slice=9)
        grid.visualize_collapse_density()
        if self.high_speed_voxel_log:
            self.plot_high_speed_voxel_log()

    def compare_clocks(self):
        plt.figure(figsize=(8,4))
        plt.plot(self.global_time_list, self.global_time_list, label="Global Clock (Sim)", linestyle="--")
        plt.plot(self.global_time_list, self.avg_local_time_list, label="Avg Local Clock")
        plt.plot(self.global_time_list, self.avg_quantum_time_list, label="Avg Quantum Clock", color="purple")
        plt.xlabel("Simulation Step")
        plt.ylabel("Clock Value")
        plt.title("Comparison of Clocks\n(Active Modes: " + ", ".join(self.modes) + ")")
        plt.legend()
        plt.show()

    def plot_high_speed_voxel_log(self):
        steps = [entry[0] for entry in self.high_speed_voxel_log]
        memories = [entry[1] for entry in self.high_speed_voxel_log]
        quantum_times = [entry[2] for entry in self.high_speed_voxel_log]
        plt.figure(figsize=(10,4))
        plt.plot(steps, memories, label="High-Speed Voxel Memory", marker="o")
        plt.plot(steps, quantum_times, label="High-Speed Voxel Quantum Time", marker="s")
        plt.xlabel("Simulation Step")
        plt.ylabel("Value")
        plt.title("Evolution of High-Speed Voxel (2,7,9)")
        plt.legend()
        plt.grid(True)
        plt.show()

# ------------------------
# 6. Main Menu
# ------------------------
def main_menu():
    print("Select modes (comma-separated). Options:")
    print("  standard       : Basic simulation")
    print("  gravity_well   : Single gravity well")
    print("  orbit          : Orbit around well center")
    print("  buffer         : Buffer qubix to counteract time dilation")
    print("  high_speed     : Assign high speed to a voxel away from center")
    print("Example: gravity_well,orbit,buffer,high_speed")
    choice = input("Enter modes: ").strip().lower()
    if not choice:
        return ["standard"]
    modes = [m.strip() for m in choice.split(",")]
    valid = {"standard","gravity_well","orbit","buffer","high_speed"}
    final_modes = [m for m in modes if m in valid]
    if not final_modes:
        print("No valid modes recognized, defaulting to standard.")
        return ["standard"]
    return final_modes

if __name__=="__main__":
    modes = main_menu()
    runner = SimulationTestRunner(
        modes=modes,
        num_steps=5000,
        grid_size=(10,10,10),
        high_speed_value=0.9
    )
    runner.run()
