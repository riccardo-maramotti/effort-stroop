## Repository overview

This repository contains the data and code used to produce the results reported in the paper:

**Maramotti R., Parr T., Tondelli M., Ballotta D., Manohar S. G., Zamboni G., and Pagnoni G.**  
*Understanding mechanisms of voluntary engagement of mental effort using active inference*  
Published in *Cognitive, Affective, and Behavioral Neuroscience*, special issue *“Neuroscience of Effort”*

The goal of the study was to investigate the mechanisms underlying the **voluntary engagement of cognitive effort**. Using a Stroop task and a computational model grounded in **active inference**, the study tested whether intentionally exerting more effort primarily operates by (i) increasing motivation for accurate performance or (ii) suppressing habitual response tendencies.

This repository provides all data, scripts, and results necessary to reproduce the analyses reported in the paper.

---

## Repository structure

### `data/`

This folder contains the experimental data collected from our sample of twenty participants.

For each participant, a dedicated subfolder is provided, containing:

- `sub_xx_taskAnswers.csv`  
  Behavioral data, including responses and reaction times.

- `sub_xx_evaluations.csv`  
  NASA-TLX subjective workload ratings.

In addition, two aggregate files are included:

- `allSubjects_demografic_data.xlsx`  
  Demographic information (age, sex) and block order for each participant.

- `allSubjects_taskAnswers.xlsx`  
  Behavioral data (responses and reaction times) from all participants combined into a single table.

---

### `results/`

This folder contains the results reported in the paper. It includes two subfolders:

- `POMDP estimates/`  
  Results of the inversion of the active inference (POMDP) model for each participant.

- `recovery analysis/`  
  Simulated data sampled from the generative model using the estimates in `POMDP estimates/` as priors. This data was used for the posterior predictive checks and parameter recovery analysis reported in the supplementary materials.

---

### `scripts/`

This folder contains all scripts required to reproduce the analyses presented in the paper. It includes two subfolders:

- `POMDP estimates/`  
  Scripts for:
  - inverting the active inference model for each participant using data from the `data/` folder;
  - performing the group-level analysis using Parametric Empirical Bayes (PEB).

- `recovery analysis/`  
  Scripts used for the posterior predictive checks and parameter recovery analysis reported in the supplementary materials.

---

## Requirements

The analyses were performed using MATLAB R2025b and the SPM25 (https://www.fil.ion.ucl.ac.uk/spm/) software package. All scripts assume that SPM25 is correctly installed and added to the MATLAB path. Moreover, the active inference model on which our analysis
is based is freely available as part of SPM25 software package (https://github.com/spm/spm/blob/main/toolbox/DEM/DEMO_MDP_Stroop.m).

---

## Citation

If you use this code or data in your work, please cite the associated paper:

```bibtex
@article{Maramotti2026,
  title   = {Understanding mechanisms of voluntary engagement of mental effort using active inference},
  author  = {Maramotti, Riccardo and Parr, Thomas and Tondelli, Manuela and Ballotta, Daniela and Manohar, Sanjay G. and Zamboni, Giovanna and Pagnoni, Giuseppe},
  journal = {Cognitive, Affective, and Behavioral Neuroscience},
  year    = {2024},
  note    = {Special Issue: Neuroscience of Effort}
}
