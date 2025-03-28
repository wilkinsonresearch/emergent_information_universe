# Quantum Clock PoC: A Classically Computable Quantum-Inspired Voxel Simulator

Quantum Clock PoC is a proof-of-concept simulation that models an emergent, quantum-inspired behavior using a classical voxel grid. This simulator demonstrates how collapse events, memory feedback, and time dilation (both gravitational and velocity-based) can interact to produce a "quantum clock" that diverges from a linear global clock—all without any quantum hardware.


## Overview

Quantum Clock PoC simulates an emergent informational universe where each voxel (or **Qubix**) behaves like a neural qubit:
- **Wavefunction Evolution:** Each voxel holds a wavefunction and undergoes phase rotation.
- **Probabilistic Collapse:** Voxels collapse based on probabilistic rules influenced by memory and local time dilation.
- **Energy and Memory Tracking:** Collapses inject energy and build up memory, which feeds back into the system.
- **Time Dilation:** The simulation incorporates gravitational time dilation (via a gravity well) and velocity-based time dilation (using special relativity concepts).
- **Multiple Modes:** The simulator supports various modes (standard, gravity_well, orbit, buffer, high_speed) for exploring different emergent phenomena.
- **Visualization:** It generates detailed visualizations such as 3D scatter plots, heatmaps, quiver plots, and clock comparisons to illustrate the dynamics.

This is a minimal and elegant engine that is entirely classically computable, and it serves as a foundation for future explorations in quantum-inspired neural models, emergent agency, and computational spacetime.


## Features

- **3D Voxel Grid:** Default grid size is 10×10×10 (configurable).
- **Wavefunction Evolution & Collapse:** Voxels evolve and collapse probabilistically, simulating quantum behavior.
- **Time Dilation:** Combines gravitational and velocity-based time dilation to update a voxel’s local time.
- **Memory Feedback:** Collapses increase voxel memory, influencing future collapse probabilities.
- **Multiple Simulation Modes:**  
  - `standard`: Baseline random behavior.  
  - `gravity_well`: Introduces a central gravitational well.  
  - `orbit`: Adds orbital dynamics around the well center.  
  - `buffer`: Uses a buffer mechanism to counteract runaway time dilation.  
  - `high_speed`: Assigns high velocity to a designated voxel (simulating a satellite).  
- **Visualization Suite:**  
  - 3D scatter plots (colored by memory or quantum time)  
  - Heatmaps for collapse density, gravitational potential, and local time dilation  
  - Velocity field quiver plots with debugging output  
  - Clock comparison plots (Global, Local, and Quantum Clocks)  
  - Time-series plots for high-speed voxel evolution


## Installation

Ensure you have Python 3 installed, then install the required dependencies:

```bash
pip install numpy matplotlib scipy
```

## After install

python simulations/quantum_clock_poc.py

## LICENSE Reminder

You're using **MIT License**, which is great:
- Permissive
- Open
- Allows use in research, startups, and remixing with attribution

## Project Structure

```bash
├── blog_code/          # Minimal demo scripts for writing/blogging
├── docs/               # Images, figures, diagrams
├── notebooks/          # Jupyter notebooks (optional)
├── simulations/        # Full simulation runs & outputs
├── LICENSE             # MIT License
└── README.md           # This file

