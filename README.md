# Threshold Boolean Formalism

This repository contains an interactive Python framework designed to simulate and analyze Gene Regulatory Networks (GRNs) using a discrete Threshold Boolean Formalism (Ising-like model). It maps out the global attractor landscape of networks under different updating rules to identify fixed-point steady states and periodic limit cycles.

## Core Formalism and Features

In this threshold logic model, every biological node (gene/protein) exists in a binary state:

* +1 (ON / Active)
* -1 (OFF / Inactive)

The regulatory state of a node $S_i$ evolves based on the net weighted sum of its inputs:
$$S_i(t+1) = \begin{cases} +1 & \text{if } \sum_{j} W_{ji} S_j(t) > 0 \\ -1 & \text{if } \sum_{j} W_{ji} S_j(t) \le 0 \end{cases}$$

### Updating Schemes

* **Synchronous Update:** All network nodes update simultaneously in a deterministic fashion. Trajectories from a given initial state follow a single fixed path.
* **Asynchronous Update:** A single node is selected at random to update at each time increment. This stochastic branching permits a single initial condition near a basin boundary to reach multiple alternative attractors (bistability/multistability).

## Outputs Generated

All analysis assets are neatly organized and exported per network topology into `./IsingResults/[Topology_Name]/`:

* **`*_sync.csv` / `*_async.csv`**: Raw state value trajectories across time steps.
* **`*_STG_*.csv`**: Edge lists and transition probabilities for building structural STGs.
* **`*_Report.png`**: Summary plot.

## Getting Started

Place your structural network topology target files inside a `./TOPOS/` folder using the standard space-separated format (`Source Target Type`).

Execute the main pipeline script or import it directly inside a Jupyter/Google Colab workspace:

```bash
python ising_sim.py
