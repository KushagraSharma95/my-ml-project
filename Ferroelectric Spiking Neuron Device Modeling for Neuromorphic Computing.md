# Ferroelectric Spiking Neuron Device Modeling for Neuromorphic Computing
**Author:** Kushagra Sharma  
**Date:** 2025-05-19  
**Status:** Working paper (GitHub preprint)  

## Abstract
Ferroelectric field‑effect transistors (FeFETs) offer non‑volatile polarization dynamics that can emulate neuronal “integrate‑and‑fire” behavior. This paper outlines compact modeling approaches for FeFET‑based spiking neurons, focusing on capture of threshold behavior, spike timing, variability, and retention. We summarize device‑to‑network co‑design principles and provide a reproducible codebase for benchmarking spiking neuron models without claiming new hardware results.

## Keywords
neuromorphic computing; spiking neurons; ferroelectric FET; compact modeling; in‑memory computing

## 1. Motivation
Neuromorphic hardware seeks brain‑like efficiency by co‑locating memory and computation and communicating via sparse spikes. Ferroelectrics are well‑suited because polarization serves as a natural state variable with hysteresis and analog update. Demonstrations of FeFET‑based spiking neurons and **all‑FeFET SNNs** show feasibility on advanced CMOS nodes, motivating accurate, lightweight neuron models.

## 2. Related work (selected)
- Editorial overview of **ferroelectric spiking neurons** and the FeFET+FET architecture.

- **All‑FeFET SNN** with on‑chip learning (surrogate gradient) in 28 nm HKMG technology.

- **Compact models** that capture FeFET neuron thresholds and spike timing for efficient SNN simulation.


## 3. Modeling approach
### 3.1 Device‑level surrogate
We implement a compact FeFET neuron model with the following components:
- Polarization state update with voltage‑dependent switching kinetics (two‑well approximation).
- Membrane‑like potential integration via polarization accumulation; threshold crossing triggers a spike and partial depolarization.
- Variability hooks for coercive voltage, retention leakage, and noise.


### 3.2 Circuit interface
- Two‑device neuron (FeFET + CMOS FET) and current‑mode readout.
- Parameterization to match measured I‑V and switching curves when available.
- Support for batch simulation of neuron arrays.


### 3.3 Network‑level use
- Surrogate‑gradient training for small benchmarks (e.g., MNIST) using the compact neuron.
- Hardware‑aware regularization to keep operating voltages and pulse counts within feasible bounds.
- Logging utilities for energy‑per‑spike estimates under simple RC assumptions.


## 4. Repository structure and reproducibility
- `models/` compact FeFET neuron; `circuits/` SPICE‑friendly wrappers; `snn/` training loops.
- Unit tests for parameter bounds; plots of switching curves and ISI distributions.
- Config‑driven experiments; results saved under `artifacts/` with JSON metadata.


## 5. Outlook
Key next steps include (i) calibrating against measured device data from published FeFET neurons, (ii) adding temperature‑dependent kinetics, and (iii) extending to synaptic FeFETs for on‑chip learning.

## 6. Suggested citation
> Sharma, K. (2025). *Ferroelectric Spiking Neuron Device Modeling for Neuromorphic Computing*. GitHub preprint. https://github.com/kushagrasharma/ferroelectric-neuron

## 7. References
1. **Ferroelectric spiking neurons** (editorial), *Nature Electronics* 2, 319–320 (2019). https://www.nature.com/articles/s41928-018-0200-3

2. Dutta, S. *et al.* “Supervised Learning in All FeFET‑Based Spiking Neural Network,” *Frontiers in Neuroscience* 14:634 (2020). https://doi.org/10.3389/fnins.2020.00634

3. Fang, Y. *et al.* “A Swarm Optimization Solver Based on Ferroelectric Spiking Neurons,” *IEEE Access* 7, 122188–122197 (2019). https://doi.org/10.1109/ACCESS.2019.2938122 (open‑access PMC)


## License
This document and code are released under **CC BY 4.0** / MIT (as applicable). See `LICENSE` files in the repo.
