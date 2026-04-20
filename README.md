# A Benchmark for Interpretable and Calibrated Risk Prediction of Steatotic Liver Disease

**CSE 6361 · University of Texas at Arlington**  
**Author:** Sachin Parajuli · `sachin.parajuli@mavs.uta.edu`

---

## Overview

This repository contains the full reproducible pipeline for a machine learning benchmark that evaluates **Logistic Regression, Explainable Boosting Machine (EBM), Random Forest, and LightGBM** on steatosis and fibrosis outcomes derived from NHANES 2017–2018 FibroScan measurements (*n* = 5,796).

Unlike most SLD prediction studies that optimize only for AUROC, this benchmark jointly evaluates:
- **Discrimination** — AUROC, AUPRC, F1
- **Calibration** — ECE, Brier score, Platt scaling, Isotonic regression
- **Interpretability** — SHAP attribution, rank stability, PDPs, EBM shape functions
- **Fairness** — Subgroup AUROC by age, sex, race/ethnicity

---

## Key Results

| Target | Best AUROC | Best Calibration (ECE) |
|--------|-----------|----------------------|
| Steatosis | RF — 0.870 | LR — 0.031 (before correction) |
| Fibrosis | LR — 0.813 | EBM — 0.036 (before correction) |

**Main finding:** Competitive AUROC does not imply well-calibrated probabilities. The EBM is the standout model for fibrosis, achieving both competitive discrimination and substantially better calibration than ensemble alternatives before any post-hoc correction.

---

## Repository Structure

```
sld-benchmark/
├── README.md
├── requirements.txt
├── sld_dataset.csv                   ← merged, labeled NHANES dataset (output of Step 1-2)
├── sld_benchmark_IEEE.tex            ← full IEEE-format paper (LaTeX)
│
├── step1_2_build_dataset.py          ← Step 1-2: merge NHANES files, create labels
├── step3_4_5_train_evaluate.py       ← Step 3-5: preprocess, train (Optuna), evaluate
├── step5_discrimination_plots.py     ← Step 5:   ROC + PR curve plots
├── step6_calibration.py              ← Step 6:   calibration metrics + reliability curves
├── step7_interpretability.py         ← Step 7:   SHAP, beeswarm, PDPs, EBM shape functions
├── step8_9_benchmark_fairness.py     ← Step 8-9: benchmark heatmap + fairness analysis
│
└── plots/
    ├── roc_curves.png
    ├── pr_curves.png
    ├── calibration_y_steatosis.png
    ├── calibration_y_fibrosis.png
    ├── shap_bar_y_steatosis_*.png    ← 4 files (one per model)
    ├── shap_bar_y_fibrosis_*.png     ← 4 files (one per model)
    ├── shap_beeswarm_y_steatosis_*.png
    ├── shap_beeswarm_y_fibrosis_*.png
    ├── pdp_y_steatosis_lgbm.png
    ├── pdp_y_fibrosis_lgbm.png
    ├── ebm_shape_y_steatosis.png
    ├── ebm_shape_y_fibrosis.png
    ├── benchmark_heatmap.png
    ├── fairness_y_steatosis.png
    └── fairness_y_fibrosis.png
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/sld-benchmark.git
cd sld-benchmark
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1–2: Build Dataset
Download the 8 NHANES 2017–2018 `.xpt` files and place them in the project directory, then run:

```bash
python step1_2_build_dataset.py
```

**Required files** (download from [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overview.aspx?BeginYear=2017)):

| File | Component |
|------|-----------|
| `DEMO_J.xpt` | Demographics |
| `BMX_J.xpt` | Body Measurements |
| `BPX_J.xpt` | Blood Pressure |
| `BIOPRO_J.xpt` | Standard Biochemistry |
| `GLU_J.xpt` | Fasting Glucose |
| `GHB_J.xpt` | Glycohemoglobin (HbA1c) |
| `ALQ_J.xpt` | Alcohol Use |
| `LUX_J.xpt` | FibroScan (LUX component) |

**Output:** `sld_dataset.csv` (already included in this repo)

---

### Step 3–5: Train Models + Evaluate Discrimination
```bash
python step3_4_5_train_evaluate.py
```
Runs Optuna Bayesian hyperparameter tuning (5-fold CV, 30 trials per model).  
⚠️ This is the slowest step — expect 30–60 min depending on your hardware.

**Output:** `splits.pkl`, `all_models.pkl`, `metrics_step5.json`

---

### Step 5: Discrimination Plots
```bash
python step5_discrimination_plots.py
```
**Output:** `plots/roc_curves.png`, `plots/pr_curves.png`

---

### Step 6: Calibration
```bash
python step6_calibration.py
```
**Output:** `calibration.pkl`, `calib_metrics.json`, `plots/calibration_*.png`

---

### Step 7: Interpretability
```bash
python step7_interpretability.py
```
**Output:** `interpretability.pkl`, all SHAP/PDP/EBM plots in `plots/`

---

### Step 8–9: Benchmark Table + Fairness
```bash
python step8_9_benchmark_fairness.py
```
**Output:** `benchmark_table.csv`, `fairness_table.csv`, `plots/benchmark_heatmap.png`, `plots/fairness_*.png`

---

## Dataset Details

| Property | Value |
|----------|-------|
| Source | NHANES 2017–2018 |
| Total participants | 5,796 (after IQR/median < 0.30 filter) |
| Features | 20 (anthropometric, biochemical, glycaemic, alcohol, demographic) |
| Steatosis prevalence | 53.3% (CAP ≥ 248 dB/m) |
| Fibrosis prevalence | 9.5% (LSM ≥ 8.0 kPa) |
| Train / Val / Test | 70% / 15% / 15% (stratified) |

---

## Models

| Model | Interpretability | Notes |
|-------|-----------------|-------|
| Logistic Regression | Intrinsic | Linear baseline |
| EBM | Intrinsic | GAM with pairwise interactions |
| Random Forest | Post-hoc (SHAP) | Robust ensemble |
| LightGBM | Post-hoc (SHAP) | Gradient-boosted trees |

---

## Dependencies

See `requirements.txt`. Key libraries:

- `scikit-learn` — preprocessing, LR, RF, calibration
- `interpret` — EBM
- `lightgbm` — LightGBM
- `optuna` — Bayesian hyperparameter tuning
- `shap` — SHAP attribution
- `imbalanced-learn` — SMOTE oversampling

---

## Paper

The full benchmark paper is available as `sld_benchmark_IEEE.tex` (IEEE conference dual-column format). Compile in [Overleaf](https://overleaf.com):
1. Upload `sld_benchmark_IEEE.tex`
2. Create a `plots/` folder and upload all PNG files into it
3. Set compiler to **pdfLaTeX**

---

## License

This project is for academic use (CSE 6361 course project). Dataset is publicly available from the CDC NHANES program.
