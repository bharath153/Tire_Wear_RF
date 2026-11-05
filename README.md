# Tire Wear Prediction — Random Forest

> Estimating remaining tread depth from synthetic vehicle and operating-condition features

## Overview

This project demonstrates a predictive-maintenance workflow for tire wear using a Random Forest regressor.

The synthetic feature set represents variables such as:
- vehicle speed and usage;
- tire load and operating conditions;
- temperature-related effects;
- road/drive-cycle influence;
- accumulated wear state.

The objective is to estimate remaining tread depth and illustrate how a data-driven model could support maintenance scheduling.

## Model

```text
RandomForestRegressor
n_estimators = 300
```

The repository reports held-out and cross-validation metrics together with feature-importance and prediction diagnostics.

## Data Scope

The dataset is **synthetic / physics-inspired** and generated for methodology demonstration. The reported accuracy is therefore not evidence of real-world fleet performance.

A deployable tire-wear estimator would require:
- measured tread-depth ground truth;
- vehicle-specific load and alignment information;
- tire compound and pressure variation;
- road-surface and weather effects;
- longitudinal validation over real usage.

## Outputs

See:
- `metrics.json` for saved evaluation metrics;
- `plots/` for prediction and analysis figures;
- `docs/` for supporting technical notes.

## Run

```bash
git clone https://github.com/bharath153/Tire_Wear_RF.git
cd Tire_Wear_RF
pip install -r requirements.txt
python src/train_rf.py
```

## Repository Structure

```text
Tire_Wear_RF/
├── src/
│   └── train_rf.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Engineering Value

The project is best read as a compact example of translating a predictive-maintenance problem into a reproducible ML workflow: feature generation, supervised regression, validation, interpretation and engineering visualization.
