# Experiment 2: Standard Mode

## Objective

The goal of this experiment is to establish a **baseline** by running the simulation **without** any special modes:
- No gravity well
- No orbit
- No buffer qubix
- No high-speed voxel

This setup provides a reference point for comparing future experiments that introduce gravitational effects, orbital motion, or high-speed dynamics.

## Setup

- **Simulation Mode:**  
  `standard` (i.e., run the code with no extra modes selected at the prompt).

- **Grid Configuration:**  
  - **Dimensions:** 10×10×10 voxels.
  - **No Gravity Well:** The center of the grid does not have increased mass or radius effect.
  - **No High-Speed Voxel:** All voxels start with zero velocity.
  - **No Buffer:** Collapses are only influenced by memory decay and base measure probability.

- **Number of Steps:**  
  Typically 2,000–5,000 steps is sufficient to observe the baseline behavior.

## Data Collection & Visualization

- **Collapse Density:**  
  Use `visualize_collapse_density` to see how collapses are distributed across the grid in the absence of gravitational or orbital influences.

- **Memory Distribution:**  
  The 3D scatter plot (`visualize_3d_grid(parameter='memory')`) will indicate how memory accumulates without any central bias.

- **Time Dilation Heatmap:**  
  In standard mode, there’s minimal or uniform time dilation. `visualize_time_dilation_heatmap` will likely show a relatively uniform distribution, influenced only by memory feedback and random collapses.

- **Comparison of Clocks:**  
  The plot showing Global Clock, Average Local Clock, and Average Quantum Clock will provide a baseline for how quickly quantum time diverges (or not) under purely random conditions.

## Expected Results

1. **Uniform Collapse Distribution**  
   - Without a gravity well or orbital effects, collapses should be **randomly** scattered. Minor variations will occur due to memory feedback, but no strong central clustering is expected.

2. **Modest Memory Gradient**  
   - Some voxels might randomly accumulate more collapses and thus slightly higher memory. However, no single region should dominate unless random chance skews the distribution.

3. **Limited Time Dilation**  
   - With no gravitational well and no high-speed voxels, time dilation arises only from local memory feedback. The **quantum clock** may still diverge from the global clock but typically **less dramatically** than in gravity or high-speed scenarios.

## Analysis & Discussion

- **Baseline Behavior:**  
  This run provides a reference for how collapses, memory, and time dilation evolve **in purely random conditions**.  
- **Comparison with Gravity Well Mode:**  
  Contrast these uniform results with the strong central clustering observed when a gravity well is active.  
- **Potential Deviations:**  
  Even in standard mode, random fluctuations can lead to local “hotspots” of collapses. Observing how memory feedback magnifies these hotspots can shed light on emergent clustering, even without gravity.

## Conclusion

Lab 2’s **standard mode** run establishes a **baseline** distribution of collapses, memory, and time dilation in the absence of gravitational or orbital effects. By comparing these uniform or random patterns with the **strong central bias** seen in the gravity well experiment, one can clearly distinguish **true gravitational influence** from **simple random fluctuations**.


