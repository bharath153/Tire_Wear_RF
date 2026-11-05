# Technical Notes — Tire Wear Random-Forest Regression

## Purpose

This repository demonstrates a supervised-learning workflow for estimating tire tread wear from synthetic vehicle/operating-condition features.

## Data

The dataset is synthetic / physics-inspired and is intended to represent factors that influence wear over use.

It is not derived from fleet CAN logs or measured tread-depth campaigns.

## Model

The implementation uses a `RandomForestRegressor` with 300 trees.

The model is suitable for this demonstration because it:
- captures nonlinear feature interactions;
- requires little feature scaling;
- provides feature-importance estimates;
- offers a strong nonlinear regression baseline.

## Evaluation

The workflow includes held-out prediction metrics and cross-validation together with diagnostic plots.

## Limitations

Real tire-wear prediction would require longitudinal measurements, tire-specific properties, inflation pressure, alignment, road surface, weather, load history and reliable tread-depth ground truth.

The current metrics should therefore be interpreted as synthetic benchmark performance, not field accuracy.
