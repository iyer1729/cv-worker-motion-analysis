# Motion Analysis Framework for Human Task Video Assessment

This repository provides implementation of a computer vision and statistics-based framework for analyzing human motion in task videos. The methods are based on:

**Hari Iyer, Neel Macwan, Shenghan Guo, and Heejin Jeong.**  
*Computer-Vision-Enabled Worker Video Analysis for Motion Amount Quantification*.  
arXiv preprint, 2024. [https://arxiv.org/abs/2405.13999](https://arxiv.org/abs/2405.13999)

The approach combines pose estimation, Hotelling’s T² anomaly detection, motion feature extraction, and regression modeling to evaluate physical workload and ergonomic risks.

---

## Repository Structure

This repository is organized into the following main folders:

- `data/`: Contains subfolders for each raw dataset:
  - `Assembly/`, `G-AI-HMS/`, `Near Duplicate/`, `UCF Sports Action/`, and `UCF50/`.
  - Each subfolder includes pose data files in `.csv` format with MediaPipe joint coordinates.

- `results/`: Contains output CSVs generated from motion analysis and statistical processing:
  - Files include `<dataset>_motion_analysis_results.csv` for each source.
  - `Merged_NASA_TLX_and_assembly_motion_data.csv` links NASA-TLX survey responses with computed motion metrics.

- `src/`: Contains the four main analysis scripts (see below).

---

## Scripts Overview

### `motion_vector_analysis.py`

Processes raw pose-estimated CSV files and computes motion vectors, motion magnitudes, and Hotelling’s T² statistics. Flags abnormal frames based on statistical control thresholds. Supports configurable step sizes and body landmark ablations.

### `clean_blank_rows.py`

Removes rows where all joint coordinates are missing or zero. Can optionally save a cleaned copy or overwrite the original. Designed for preprocessing MediaPipe pose output.

### `rf_cross_dataset_warning_classifier.py`

Trains a Random Forest model on the Assembly dataset to classify motion warnings and evaluates generalization to other datasets. Uses adaptive per-dataset thresholds for labeling warnings based on Hotelling’s T² values.

### `tlx_motion_statistical_analysis.py`

Performs statistical evaluation of motion data and subjective workload ratings (NASA-TLX). Includes:
- Descriptive statistics
- Pearson correlation matrix
- Pairwise t-tests across ablation groups
- Linear regression to predict motion warnings
- Random Forest regression including categorical ablation
- Pairwise correlations between TLX subscales and motion warnings

---

## Key Results

This repository supports the experimental framework and results presented in [Iyer et al., 2024](https://arxiv.org/abs/2405.13999).

- **Motion magnitude** was higher in large-object tasks (mean = 0.097 at step 0), but small-object tasks showed greater motion *variability* and more sensitivity to fine movements.
- **RMSD** for small-object (S) tasks: 0.072 meters, compared to 0.035 meters for large-object (L) tasks.
- **Correlation between motion amount and Hotelling’s T²** was 35% higher for S tasks, indicating better alignment between statistical monitoring and observed micromovements.
- **Hotelling’s T² warnings** peaked in L tasks (e.g., 77 warnings for LGP at step 4), showing high control-limit violations due to coarse joint motions.
- **Random Forest classifiers** achieved over 85% accuracy when trained on Assembly data and applied to unseen datasets using only four motion features (RMSD, mean_motion, std_motion, UCL).
- **NASA-TLX metrics** like Physical Demand, Effort, and Mental Demand correlated (_r_ = 0.218, _p_ < 0.005) with computed joint motion indicators.

---

## Citation

If you use this repository or adapt any of its methods or scripts, please cite:

```bibtex
@misc{iyer2024computer,
  title={Computer-Vision-Enabled Worker Video Analysis for Motion Amount Quantification},
  author={Hari Iyer and Neel Macwan and Shenghan Guo and Heejin Jeong},
  year={2024},
  eprint={2405.13999},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
