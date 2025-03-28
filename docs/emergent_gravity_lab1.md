# Experiment 1: Gravity’s Influence on Collapse Placement

## Objective

Investigate how a centrally located gravitational well biases collapse events and memory buildup within the voxel grid. This experiment simulates how a massive object curves spacetime (as in general relativity), leading to higher collapse rates and increased memory in regions near the gravity well.

## Setup

- **Simulation Mode:**  
  Use the `gravity_well` mode only to activate a single gravity well.

- **Grid Configuration:**  
  - **Dimensions:** 10×10×10 voxels.  
  - **Gravity Well Parameters:**  
    - **Center:** Middle of the grid (e.g., `(5,5,5)` or adjusted as needed).  
    - **Radius:** 3 grid units.  
    - **Mass & Effect:** Example values: `mass = 1e3`, `effect = 0.5`.

- **Simulation Duration:**  
  Run the simulation for 2000–5000 steps to accumulate sufficient data.

## Data Collection & Visualization

- **Gravitational Potential Heatmap:**  
  - Use the `visualize_gravitational_potential` method to generate a 2D heatmap (for a specific z-slice) showing the gravitational potential field.
  - **Expected:** A clear dip or gradient at the center where the mass is concentrated.

- **Collapse Density Analysis:**  
  - The simulation tracks the number of collapses per voxel.
  - Use the `visualize_collapse_density` method to plot a heatmap of collapse counts.
  - **Expected:** Higher collapse densities near the gravity well.

- **Memory Gradient Visualization:**  
  - Use the `visualize_3d_grid(parameter='memory')` function to create a 3D scatter plot of voxels colored by their memory value.
  - **Expected:** Voxels near the gravity well should have higher memory values.

- **Time Dilation Heatmap:**  
  - Use the `visualize_time_dilation_heatmap` method (on a selected z-slice) to view the distribution of local time dilation.
  - **Expected:** Areas under stronger gravitational influence should exhibit higher dilation.

## Expected Results

- **Collapse Concentration:**  
  Voxels within the gravity well’s influence should collapse more frequently, leading to higher collapse densities.

- **Memory Buildup:**  
  Memory values should be highest near the center of the gravity well, indicating a positive feedback loop.

- **Time Dilation Gradient:**  
  The heatmap should reveal a gradient where local time is most “stretched” near the gravity well.

## Analysis & Discussion

- **Comparison with Theory:**  
  Compare the gravitational potential heatmap with the collapse density and memory gradient. Validate whether the emergent behavior aligns with the weak-field GR predictions.

- **Parameter Sensitivity:**  
  Experiment with adjustments to gravity well parameters (mass, radius, effect) to see how they affect the simulation output.

- **Implications:**  
  Discuss how these patterns might reflect real-world gravitational influences on time and structure formation.

## Example Code Snippet

Below is an example code block (in Python) that illustrates key parts of the simulation setup:

```python
class SimulationConfig:
    def __init__(self):
        self.speed_of_light = 1.0
        self.memory_alpha = 0.5
        self.collapse_beta = 0.3
        self.base_measure_prob = 0.10
        self.memory_decay_rate = 0.99
        self.memory_increase_factor = 0.5
        self.G = 1e-3
        self.M = 1e3

class Qubix:
    def __init__(self, q_id, position, config=None):
        self.id = q_id
        self.position = position
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

    def update_memory(self, did_collapse):
        if did_collapse:
            self.memory += self.config.memory_increase_factor
        self.memory *= self.config.memory_decay_rate

