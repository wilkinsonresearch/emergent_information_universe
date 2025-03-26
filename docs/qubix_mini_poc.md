# Qubix: A Classically Computable Quantum-Inspired Voxel Simulator

> A minimal and elegant simulation of emergent quantum-like behavior using classical computation. No quantum hardware required.

## Overview

**Qubix** is a voxel-based field simulation where each unit behaves like a "neural qubit":
- Holds a wavefunction (`ψ = α|0> + β|1>`)
- Evolves by phase rotation
- Collapses probabilistically (or by condition)
- Tracks energy decay and memory growth
- Behaves collectively like a quantum field or cognitive substrate

This is a **proof-of-concept** engine for future directions in:
- Quantum-inspired neural models
- Emergent agency
- Computational spacetime

## Features

- 3D voxel grid (default: 5×5×5)
- Wavefunction evolution and collapse
- Energy tracking and memory growth
- Entropic decay and visualization
- Animated GIFs for memory and energy slices
- Fully classically computable

## Install

pip install numpy matplotlib imageio natsort

## After install

python simulations/qubix_energy_sim.py

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

