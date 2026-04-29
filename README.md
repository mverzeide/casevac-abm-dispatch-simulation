# CASEVAC Simulation in Dynamic Contested Environments

This repository contains Python code for an agent-based simulation study of military casualty evacuation (CASEVAC) in dynamic contested environments. The project was developed for the MSc thesis:

**Military Casualty Evacuation in Dynamic Contested Environments: A Simulation Study of Dispatch Heuristics**  
Marijn Verzeide, Tilburg University, 2026

## Project overview

The simulation models CASEVAC as a dynamic operational system in which casualties, evacuation platforms, accessibility constraints, and treatment-base capacity evolve over time. The model is designed to evaluate how different dispatch heuristics perform when evacuation decisions must be made under uncertainty.

The framework includes:

- stochastic casualty generation;
- dynamic no-go areas and time-varying road accessibility;
- heterogeneous evacuation platforms, including ambulances and helicopters;
- casualty triage and deterioration over time;
- treatment-base queueing and bed-capacity constraints;
- mission interruption under operational doctrine assumptions;
- rule-based dispatch heuristics for comparing evacuation policies.

The central research question is:

> Under which operational conditions do dispatch heuristics significantly affect casualty evacuation outcomes?

The main conclusion of the thesis is that no dispatch heuristic is universally optimal. Dispatch performance depends on the outcome metric, fleet composition, capacity regime, and operational doctrine. Severity-based dispatch tends to perform better for mortality-related outcomes under stronger scarcity, while proximity-based ambulance assignment can perform better for timely treatment-base arrival in several helicopter-enabled settings.

## Repository structure

```text
.
├── CASEVAC_MVERZEIDE_VISUALIZATION.py   # Single-run simulation and visualization script
├── CASEVAC_MVERZEIDE_MONTECARLO.py      # Monte Carlo experiment script
├── CASEVAC_MVERZEIDE_MAP_DOWNLOADER.py  # OpenStreetMap road-network downloader
├── gulpen_polygon_simplified.graphml    # Generated road-network input file
└── README.md
```

### `CASEVAC_MVERZEIDE_VISUALIZATION.py`

Runs a CASEVAC simulation scenario and visualizes the resulting environment, casualties, platforms, and evacuation dynamics. This file is intended for inspecting model behavior, debugging, and producing illustrative simulation output.

### `CASEVAC_MVERZEIDE_MONTECARLO.py`

Runs repeated simulation experiments for comparing dispatch heuristics under different random seeds and scenario settings. This file is intended for batch experimentation and robustness checks.

### `CASEVAC_MVERZEIDE_MAP_DOWNLOADER.py`

Downloads a road network from OpenStreetMap using OSMnx and saves it as a GraphML file. The default configuration downloads the Gulpen area and creates:

```text
gulpen_polygon_simplified.graphml
```

This generated file is required by the simulation scripts.

## Model components

The simulation contains the following main components:

| Component | Description |
|---|---|
| Casualties | Agents with location, triage status, deterioration schedule, assignment status, and evacuation status. |
| Ambulances | Ground platforms that move through the safe road network and complete a final off-road approach to casualties. |
| Helicopters | Air platforms that move in continuous space subject to straight-line no-go feasibility. |
| Coordinator | Central myopic dispatcher that assigns available platforms to eligible casualties. |
| Treatment base | Fixed treatment location with limited beds and queueing logic. |
| Environment | Dynamic conflict area with changing no-go tiles, fight intensity, situation dynamics, and casualty hotspots. |

## Dispatch heuristics

The model compares six baseline dispatch heuristics:

| ID | Ambulance selection | Helicopter selection | Dispatch order |
|---:|---|---|---|
| 1 | Severity → nearest | Severity → farthest by ambulance travel time from base | Ambulance first |
| 2 | Nearest only | Farthest only by ambulance travel time from base | Ambulance first |
| 3 | Severity → nearest | Severity → farthest by ambulance travel time from base | Helicopter first |
| 4 | Nearest only | Farthest only by ambulance travel time from base | Helicopter first |
| 5 | Nearest only | Severity → farthest by ambulance travel time from base | Ambulance first |
| 6 | Nearest only | Severity → farthest by ambulance travel time from base | Helicopter first |

Black casualties are excluded from dispatch. Casualties can deteriorate over time if they are not picked up or treated quickly enough.

## Operational doctrines

The model supports two doctrine settings for platform exposure to no-go areas:

| Doctrine | Behavior |
|---|---|
| `HARD` | Platform exposure to no-go areas results in permanent platform loss. |
| `SOFT` | Platform exposure triggers mission abort and retreat, but the platform remains operational. |

## Key outcome metrics

The simulation records metrics such as:

- total casualties generated;
- casualties picked up;
- casualties delivered to the treatment base;
- casualties arriving within the Golden Hour;
- casualties deteriorating to black;
- deaths in queue;
- black-on-arrival cases;
- platform failures;
- mission aborts;
- base queue length and bed utilization;
- no-go and reachability failure counts.

## Requirements

The code uses the following main Python packages:

```text
agentpy
osmnx
networkx
matplotlib
numpy
scipy
shapely
```

A typical installation command is:

```bash
pip install agentpy osmnx networkx matplotlib numpy scipy shapely
```

Depending on your local Python environment, OSMnx may require additional geospatial dependencies. Using a clean Conda environment is recommended for easier installation of geospatial packages.

## Input data and map download

The simulation scripts expect the road-network file below to be available in the working directory:

```text
gulpen_polygon_simplified.graphml
```

This file can be generated with the included map downloader:

```bash
python CASEVAC_MVERZEIDE_MAP_DOWNLOADER.py
```

By default, the downloader uses the `Gulpen` configuration and saves the road network as `gulpen_polygon_simplified.graphml`. The downloader retrieves road data from OpenStreetMap through OSMnx, simplifies the road-network topology, adds speed and travel-time attributes, and saves the result as a GraphML file.

To download another region, edit the `CITY` variable in `CASEVAC_MVERZEIDE_MAP_DOWNLOADER.py`:

```python
CITY = "Gulpen"
```

Available built-in options are:

```text
Gulpen, Bergen, Valkenburg, Lapland, Almelo, Houten, Sneek, Varėna, Nicosia
```

Make sure that the `GRAPH_FILE` produced by the downloader matches the `GRAPH_FILE` expected by the simulation scripts. The default simulation configuration expects:

```python
GRAPH_FILE = "gulpen_polygon_simplified.graphml"
```

The simulation loads this file with OSMnx and projects it before execution. Edge travel times are computed from edge length and speed attributes. If speed information is missing, a default speed is used.

## How to run

### Download the map

First generate the required GraphML road-network file:

```bash
python CASEVAC_MVERZEIDE_MAP_DOWNLOADER.py
```

After this step, `gulpen_polygon_simplified.graphml` should be present in the repository folder.

### Single-run visualization

```bash
python CASEVAC_MVERZEIDE_VISUALIZATION.py
```

Use this script to inspect one scenario visually and to generate simulation traces or animations.

### Monte Carlo experiments

```bash
python CASEVAC_MVERZEIDE_MONTECARLO.py
```

Use this script to run repeated experiments. The provided file represents one Monte Carlo run configuration. In the thesis experiments, multiple versions were run with different seeds.

## Important configuration parameters

The most important parameters are defined near the top of the scripts:

```python
seed = 52
DISPATCH_HEURISTIC = 6
USE_CLAIRVOYANT_COORDINATOR = False
Base_Policy = "SEVERITY"
PLATFORM_DOCTRINE = "HARD"
TIME_STEPS = 300
```

Other parameters control casualty generation, triage probabilities, deterioration delays, hotspot behavior, map resolution, platform speeds, platform capacities, and treatment-base capacity.

## Reproducibility notes

The model uses random seeds for reproducibility:

```python
random.seed(seed)
np.random.seed(seed)
```

For Monte Carlo analysis, run the experiment multiple times with different seed values and aggregate the resulting metrics externally or by extending the batch logic in the script.

The simulation is synthetic and scenario-based. It is intended as an experimental framework for comparing dispatch logic under controlled stochastic conditions, not as a calibrated operational forecasting model.

## Research interpretation

The thesis results support a conditional view of dispatch effectiveness:

- dispatch logic matters most when evacuation capacity is scarce and prioritization remains binding;
- severity-based prioritization is often stronger for mortality-related outcomes;
- proximity-based dispatch can improve timely arrival outcomes, especially in some helicopter-enabled regimes;
- performance differences between heuristics tend to shrink as capacity expands;
- system-level constraints such as accessibility loss, fleet scarcity, and treatment congestion can dominate dispatch-policy effects.


## Disclaimer

This repository is intended for academic research and simulation-based analysis. The model is a stylized representation of CASEVAC dynamics and should not be used as an operational decision-support system without validation, calibration, and domain-expert review.
