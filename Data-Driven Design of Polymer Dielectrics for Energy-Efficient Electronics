# Data-Driven Design of Polymer Dielectrics for Energy-Efficient Electronics
**Author:** Kushagra Sharma  
**Date:** 2025-03-14  
**Status:** Working paper (GitHub preprint)  

## Abstract
Polymer dielectrics are central to energy storage and insulation in modern electronics. Yet common materials (e.g., BOPP) struggle to combine high energy density with thermal robustness, limiting use in harsh environments. This working paper outlines an informatics workflow—data assembly, representation, supervised learning, and inverse design—to accelerate the discovery of polymer dielectrics that sustain high energy density at high temperature. The paper also describes an open, reproducible repository structure to support future experiments and comparisons.

## Keywords
polymer dielectrics; materials informatics; high-temperature capacitors; inverse design; active learning

## 1. Motivation and background
Electrostatic capacitors rely on the dielectric to set their **energy density** and temperature limits. In practice, widely used polymer dielectrics (e.g., BOPP) provide excellent breakdown strength but lose performance at elevated temperatures, motivating new chemistries that unite **high permittivity**, **thermal stability**, and **mechanical integrity**. Recent AI-assisted discovery identified polynorbornene- and polyimide-based dielectrics that reach **~8.3 J/cc at 200 °C**, about **11×** higher than any commercial polymer dielectric at that temperature—an example of how data and molecular engineering can overcome longstanding trade-offs.


**Pointers for readers:** see the references for overviews of Polymer Genome, polymer informatics, and high-temperature dielectric design. This repo is designed to plug into those efforts and provide a clean baseline for future studies.


## 2. Problem statement
We seek polymer candidates that maximize **energy density** at elevated temperature while respecting constraints on **loss**, **breakdown strength**, **processability**, and **safety**. Because the chemical design space is enormous, brute-force synthesis is infeasible. Instead, we build predictive models and couple them with search strategies to prioritize a small, testable set of candidates.

## 3. Methods
### 3.1 Data
- **Sources.** Public polymer datasets and literature-derived measurements (dielectric constant, band gap, glass transition, loss tangent, breakdown strength). Where exact values are unavailable, the code provides placeholders and parsing utilities for user-supplied data tables.
- **Curation.** Deduplicate by canonical SMILES; standardize units and temperature/frequency conditions; record provenance in `data/manifest.csv`.
- **Features.** (i) Structure-based embeddings (SMILES tokenization and chemical language models, e.g., polyBERT); (ii) physics-inspired features (electronegativity counts, ring/saturation descriptors); (iii) simple text features from literature for conditions (temperature, frequency).


### 3.2 Models
- **Property prediction.** Gradient boosting and calibrated linear models for permittivity and loss at specified conditions; uncertainty via ensembling and conformal prediction.
- **Breakdown surrogate.** Conservative classifier trained on positive/negative breakdown reports (with data augmentation from DFT-derived band gaps where justified).
- **Multi-objective ranker.** Scalarizes targets (e.g., energy density proxy at T, loss ceiling) under uncertainty penalties.


### 3.3 Search and design
- **Sequential (active) learning.** Iteratively propose candidates that are high-value and high-uncertainty to reduce posterior error with minimal experiments.
- **Genetic/inversion loops.** Mutate/compose monomers under synthetic rules; filter with property and safety constraints.
- **Human-in-the-loop.** Notebooks produce ranked shortlists with provenance to aid experimental selection.


### 3.4 Reproducibility
- `environment.yml` with pinned versions; `Makefile` targets for `lint`, `test`, `train`, `rank`.
- MLflow tracking for datasets, parameters, metrics, and artifacts.
- Unit tests for featurization and target alignment; schema checks for data drops.


## 4. Case study (illustrative)
We provide notebooks that reproduce a **toy screening** using public data to demonstrate the pipeline. The notebooks stop short of claiming new state-of-the-art; they exist to validate the code path and to be extended with lab data by collaborators.

## 5. Discussion and next steps
- Expand curated datasets (temperature/frequency-resolved permittivity and loss; breakdown statistics with confidence).
- Add domain constraints learned from recent high-temperature dielectric successes (e.g., backbone rigidity, fluorination motifs, thermal transitions).
- Integrate device-level simulations to bridge material predictions with capacitor performance.


## 6. Ethical and safety considerations
- Respect licenses on literature-derived data; include citations and links.
- Avoid disclosing proprietary formulations; provide aggregates when NDAs apply.

## 7. How to use this repository
```bash
# create env and run tests
conda env create -f environment.yml
conda activate polymer-diel
make test

# train models and generate ranking
make train
make rank
```
Artifacts (trained models, ranked candidates, and reports) will be saved under `artifacts/` with MLflow metadata.

## 8. Suggested citation
> Sharma, K. (2025). *Data-Driven Design of Polymer Dielectrics for Energy-Efficient Electronics*. GitHub preprint. https://github.com/kushagrasharma/polymer-dielectric-ML

## 9. References
1. Gurnani, R. *et al.* “AI-assisted discovery of high-temperature dielectrics for energy storage,” *Nature Communications* (2024). https://doi.org/10.1038/s41467-024-50413-x

2. Kim, C. *et al.* “A Data-Powered Polymer Informatics Platform for Property Prediction,” *J. Phys. Chem. C* 122, 17575–17585 (2018). https://doi.org/10.1021/acs.jpcc.8b02913

3. Kuenneth, C. *et al.* “polyBERT: a chemical language model to enable fully machine-readable polymers,” *Nature Communications* 14, 3774 (2023). https://doi.org/10.1038/s41467-023-39868-6

4. Chen, L. *et al.* “Dielectric Polymers Tolerant to Electric Field and Temperature Extremes,” *ACS Appl. Mater. Interfaces* 13, 35704–35716 (2021). https://doi.org/10.1021/acsami.1c11885

5. Chandrasekaran, A. *et al.* “Polymer Genome: A Polymer Informatics Platform…,” *Lecture Notes in Physics* 968, 397–412 (2020). https://doi.org/10.1007/978-3-030-40245-7_16

6. Ramprasad Group resources (Polymer Genome, datasets, and tools): https://ramprasad.mse.gatech.edu/ and https://khazana.gatech.edu/


## License
This document and accompanying code are released under **CC BY 4.0** / MIT (as applicable). See `LICENSE` files in the repo.
