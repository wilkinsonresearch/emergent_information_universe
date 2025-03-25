# Emergent Informational Universe Simulation

emergent_information_simulation.py

A toy simulation demonstrating an emergent informational universe through hybrid classical collapse events, memory feedback, and repulsive vector field dynamics. This project provides a proof-of-concept for how local computational events (collapses) can reshape an underlying probability/energy field and give rise to emergent phenomena such as directed flows, stabilizing vacuums, and information release.

## Overview

This simulation models a real-valued grid where:
- **Collapse events** occur in random 3×3 regions, injecting energy and updating a memory field.
- **Energy diffusion and dissipation** shape the evolving field.
- A **vector field** is computed from local energy gradients and enhanced with a repulsion mechanism (to mimic pressure release).
- The emergent behavior is visualized using an animation with four subplots:
  - The quantum grid (with collapses).
  - A quiver plot of the vector field.
  - A memory feedback (scaffold) heatmap.
  - A metrics plot showing the evolution of entropy and total energy.

This is intended as a modular, extensible “toy” demonstration that may serve as a basis for further research or additional simulation modules (e.g., interference, cellular automata, or diffusion-limited aggregation).

## Features

- **Hybrid Model:** Combines local collapse events with energy diffusion and memory feedback.
- **Repulsive Vector Field:** Implements a repulsion mechanism to avoid runaway buildup and encourage space-filling flows.
- **Dynamic Visualization:** Animated display of grid, vector field, memory, and metrics.
- **Post-Simulation Analysis:** PCA and clustering of snapshots for further insights.
- **Modular Code Structure:** Easy to extend and integrate additional simulations.

## Requirements

This project requires Python 3.8 or higher along with the following packages:

- numpy
- matplotlib
- torch
- scikit-learn
- scipy

## Installation

It is recommended to use a virtual environment.
```bash
python3.10 -m venv venv
source /folder/venv/bin/activate
```
Then, install the dependencies using pip:
```
bash
pip install numpy matplotlib torch scikit-learn scipy
```
## After installing

After installing the required packages, you can run the simulation from the command line:

python emergent_information_simulation.py

This command will:

    Generate an animated GIF (e.g., emergent_information_simulation.gif)

    Save snapshots of the simulation (in the data/ folder)

    Perform post-simulation analysis (PCA, clustering, etc.) and save the results as files
