# CMAPSS Predictive Maintenance (NASA)

This project implements a clean, reproducible baseline pipeline for **Remaining Useful Life (RUL)** prediction using NASA’s CMAPSS turbofan engine dataset (FD001).

The focus is on **sound data handling, target engineering, evaluation, and uncertainty awareness**.

## Problem Description

Each engine is monitored through multiple sensors over operational cycles until failure.

The goal is to predict **Remaining Useful Life (RUL)** — the number of cycles remaining before engine failure — from sensor data.

This is a **predictive maintenance** problem.

## Data Processing

- Raw CMAPSS data is loaded directly from NASA text files.
- RUL is computed as:

```
RUL = max_cycle_per_engine − current_cycle
```


### RUL Capping

To reflect uncertainty in early-life degradation, RUL values are capped:

- **RUL > 125 → RUL = 125**

This follows common CMAPSS practice and avoids over-weighting early-life samples where sensor signals are weak.

## Feature Handling

- Constant (non-informative) sensors are automatically detected and removed.
- Remaining operational settings and sensor values are used as model inputs.
- Data is standardized using `StandardScaler`.

## Modeling Approach

A **baseline linear regression model** is used to establish a transparent and interpretable reference.

- Train/test split is performed **by engine ID**, preventing information leakage across engine lifetimes.
- Evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

## Results

### Overall Performance (FD001)

- **MAE:** ~14–18 cycles
- **RMSE:** ~18 cycles

(Exact values vary with train/test split.)

### Error Analysis by Life Stage

Model performance differs across engine life stages:

- **Late life (RUL ≤ 20):** lower error, stronger degradation signal
- **Mid life (20 < RUL ≤ 60):** highest uncertainty
- **Early life (RUL > 60):** moderate error due to weak degradation patterns

This behavior is consistent with real-world degradation processes.

## Uncertainty Considerations

To assess model sensitivity to data variability, training is repeated across multiple random **engine-level splits**.

- MAE across 10 runs:
  - **Mean:** ~16.7 cycles
  - **Std:** ~0.9 cycles

Prediction variability across splits is used as a **simple uncertainty proxy**, highlighting regions where predictions are less stable.

## Key Takeaways

- Careful target engineering and evaluation are critical for predictive maintenance.
- Even simple baseline models can provide meaningful insights when combined with:
  - proper data splits
  - life-stage-aware analysis
  - uncertainty considerations

## Tools

- Python, Pandas, NumPy
- scikit-learn
- Git / GitHub

## Dataset

NASA CMAPSS Turbofan Engine Degradation Simulation Dataset
