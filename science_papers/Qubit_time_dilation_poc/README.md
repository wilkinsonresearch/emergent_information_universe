# Qubix Time Dilation Simulation

> A minimal and elegant simulation of emergent quantum collapse behavior under relativistic time dilation — using fully classical computation.

---

## Overview

**Qubix Time Dilation** simulates a 3D voxel field where each unit behaves like a locally timed collapse candidate:
- Subject to **gravitational and/or velocity-based time dilation**
- Evolves and collapses probabilistically in a **quantum-like** fashion
- Accumulates **memory saturation** over time
- Outputs visualizations reflecting structural emergence and decoherence-like behavior

This is a **proof-of-concept model** for exploring:
- Emergent decoherence-like behavior
- Collapse field geometry
- Time-asymmetric computation
- Information-dense structures under relativistic distortion

---

## Features

- Voxel-based simulation with local collapse clocks
- Time dilation (GR and SR) per voxel
- Per-voxel memory tracking and saturation dynamics
- Mode-separated simulations (gravity, velocity, combined)
- Animated GIFs of collapse progression
- Heatmaps of time dilation and collapse density
- 3D plots of memory and quantum time
- CSV export of collapse data
- Full simulation logging with timestamp
- Summary CSV with per-mode timing stats
- Jupyter-friendly + headless CLI support

---

## Modes

| Mode      | Description                             |
|-----------|-----------------------------------------|
| `standard`| No time dilation (baseline)             |
| `gravity` | Gravitational time dilation only        |
| `velocity`| Velocity-based (SR) time dilation only  |
| `combined`| Both gravitational and SR effects       |

---

## Installation

Install dependencies:

```bash
pip install numpy matplotlib imageio ipywidgets pyqt5
```

Jupyter support (optional):

```bash
pip install notebook
pip install plotly
```

---

## Usage

To run all modes and save outputs:

```bash
python Qubix_poc.py
```

To explore interactively in Jupyter:

```bash
jupyter notebook notebooks/Qubix_poc.ipynb
```

---

## Output Files

For each mode, you get:

| File                          | Description                             |
|-------------------------------|-----------------------------------------|
| `collapse_memory.gif`         | Animated collapse heatmap over time     |
| `qubix_collapse.csv`          | Per-voxel collapse timing data          |
| `collapse_density.png`        | Collapse frequency heatmap (Z-slice)    |
| `3d_memory.png`               | Final 3D memory state (colored)         |
| `time_dilation_heatmap.png`   | Global vs. Local vs. Quantum clock plot |
| `frames/frame_[nr].png`       | Per-frame collapse heatmap              |

Plus shared files:

| File                           | Description                            |
|--------------------------------|----------------------------------------|
| `time_comparison_all_modes.png`| Avg local time vs global across modes  |
| `final_memory_histogram.png`   | Memory histogram                       |
| `simulation_log.txt`           | Full console log of simulation         |

---

## Jupyter Notebook

Use the interactive notebook to:
- Run a single mode with rich display
- Tweak gravity, velocity, resolution, and probability
- Explore voxel-level effects across space and time

Notebook:
📄 `notebooks/Qubix_poc.ipynb`

---

## Project Structure

```
QubixTimeDilation/
├── Qubix_poc.py           # Full simulation engine (headless + log)
├── notebooks/
│   └── Qubix_poc.ipynb    # Interactive exploration in Jupyter
├── outputs/
│   ├── standard/                  # Output files per mode
│   ├── gravity/
│   ├── velocity/
│   ├── combined/
│    ├── simulation_log.txt            # Full console output
│    └── summary_stats.csv             # One-line summary per mode
├── README.md
├── requirements.txt
└── LICENSE
```

---

## License

MIT License — free to use, modify, cite.

---

> Research direction by Wilkinson Research
> Simulation by Keijo Wilkinson 
> keijo.wilkinson@gmail.com
> https://WilkinsonResearch.se
