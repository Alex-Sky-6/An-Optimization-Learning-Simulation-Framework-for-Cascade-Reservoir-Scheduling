# An Optimization-Learning-Simulation Framework for Cascade Reservoir Scheduling

## Project Overview

This repository contains the core algorithm files for an optimization-learning-simulation framework for cascade reservoir scheduling. The framework is organized around three complementary components:

- **Optimization**: a MATLAB cascade-reservoir scheduling problem definition for multi-objective reservoir operation.
- **Learning**: a PyTorch LSTM predictor with DQN-based adaptive loss weighting and physical-constraint loss terms.
- **Simulation**: MATLAB SARIMA utilities for hydrological time-series forecasting and comparison.

The current public repository provides the main algorithmic scripts only. Engineering datasets, trained model artifacts, and some local framework dependencies are not included because of data security and engineering-parameter confidentiality.

## Repository Structure

```text
.
|-- Optimization/
|   `-- reservoir.m            # MATLAB cascade reservoir optimization problem definition
|-- Learning/
|   |-- wddlearning.py          # LSTM + DQN training script with physical-constraint loss
|   |-- wddinference_test.py    # Lightweight inference API for saved PyTorch model artifacts
|   `-- runwdd.py              # Example inference script
|-- Simulation/
|   |-- Fun_SARIMA_Forecast.m  # SARIMA forecasting workflow
|   |-- SARMA_Order_Select.m   # SARIMA order selection using AIC/BIC
|   `-- creatSARIMA.m          # SARIMA model construction helper
`-- README.md
```

## Component Details

### 1. Optimization Module

The optimization module is implemented in MATLAB:

```text
Optimization/reservoir.m
```

`reservoir.m` defines a cascade reservoir scheduling problem class derived from `PRORES`. It includes:

- monthly water-level decision variables for four cascade reservoirs;
- reservoir-specific upper and lower water-level bounds;
- initial water levels;
- discharge constraints;
- power output constraints;
- monthly inflow loading;
- objective calculation for:
  - hydropower generation,
  - flood-control and water-supply regulation,
  - ecological/environmental operation indicators;
- constraint violation calculation for water level, discharge, and power output.

This file depends on the external optimization framework that provides the `PRORES` base class and related runtime environment.

### 2. Learning Module

The learning module is implemented in Python under `Learning/`.

Main script:

```text
Learning/wddlearning.py
```

It includes:

- data loading and normalization with `pandas` and `MinMaxScaler`;
- sequence construction with a default window length of 3 time steps;
- an LSTM predictor (`LSTMPredictor`);
- a DQN agent (`DQNAgent`) for dynamic weighting between prediction loss and physical loss;
- physical-constraint loss terms, including non-negativity, water balance, discharge bounds, operation-cycle smoothness, and storage-capacity bounds;
- final evaluation metrics: MAE, MSE, RMSE, R2, NSE, KGE, and WBI;
- model/scaler output files:
  - `model_best.pt`
  - `scaler_input.pkl`
  - `scaler_target.pkl`
  - `Complete_Prediction_Results_Training_and_Test.xlsx`

Inference scripts:

```text
Learning/wddinference_test.py
Learning/runwdd.py
```

`wddinference_test.py` defines `ModelAPI`, which loads the trained model and scalers and supports:

- `predict_next_step(df)` for one-step prediction;
- `rolling_predict(df, steps=12)` for rolling multi-step prediction.

Before running the learning scripts, update the local data paths, column names, and reservoir physical parameters to match your dataset.

### 3. Simulation Module

The simulation module is implemented in MATLAB under `Simulation/`.

The main entry point is:

```text
Simulation/Fun_SARIMA_Forecast.m
```

It performs:

- automatic differencing-order testing using ADF and KPSS tests;
- ACF/PACF diagnostics;
- SARIMA order selection through `SARMA_Order_Select.m`;
- model creation through `creatSARIMA.m`;
- parameter estimation;
- residual diagnostics;
- multi-step forecasting;
- 95% confidence interval calculation.

Example MATLAB usage:

```matlab
addpath(genpath('Simulation'));

data = readmatrix('your_monthly_series.xlsx');
step = 12;
max_ar = 3;
max_ma = 3;
max_sar = 2;
max_sma = 2;
season = 12;
figflag = 'on';

[forecast, lower, upper] = Fun_SARIMA_Forecast( ...
    data, step, max_ar, max_ma, max_sar, max_sma, season, figflag);
```

## Requirements

### Python

Recommended Python dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn torch joblib openpyxl
```

The learning module was written for CPU execution by default. CUDA can be enabled in the inference API if a compatible PyTorch/CUDA environment is available.

### MATLAB

Recommended MATLAB environment:

- MATLAB R2016b to R2020, based on the compatibility branches in the SARIMA script;
- Econometrics Toolbox;
- Statistics and Machine Learning Toolbox;
- the external optimization framework that defines `PRORES`, if running `Optimization/reservoir.m`.

## Basic Usage

### Use the Optimization Problem

Place `Optimization/reservoir.m` in the MATLAB path together with the required optimization framework and input data file, then instantiate or call it through the framework that provides `PRORES`.

`reservoir.m` currently expects the monthly runoff data file:

```text
Runoff-Wudongde-Monthly-Data-Standardized.xlsx
```

Update the file name, year selection, and data-column mapping as needed for your experiment.

### Train the Learning Model

```bash
cd Learning
python wddlearning.py
```

Before training, check these items in `wddlearning.py`:

- `filepath`: path to the training Excel file;
- `input_features`: input column names;
- `target_feature`: prediction target column;
- physical parameters used by `physical_constraint_loss`, such as discharge and storage bounds.

### Run Inference

After training produces `model_best.pt`, `scaler_input.pkl`, and `scaler_target.pkl`, use:

```python
import pandas as pd
from wddinference_test import ModelAPI

api = ModelAPI()
df = pd.read_excel("your_inference_data.xlsx")

next_value = api.predict_next_step(df)
future_values = api.rolling_predict(df, steps=12)
```

Make sure the inference DataFrame contains the columns listed in `INPUT_FEATURES` in `wddinference_test.py`.

### Run SARIMA Forecasting

```matlab
addpath(genpath('Simulation'));
data = readmatrix('your_monthly_series.xlsx');
[forecast, lower, upper] = Fun_SARIMA_Forecast(data, 12, 3, 3, 2, 2, 12, 'on');
```

## Notes for Reproducibility

- Random seeds are set in the Python training script for `numpy`, `torch`, and `random`.
- Local absolute data paths in the scripts are placeholders from the original experiment environment and should be replaced before reuse.
- The repository does not include confidential datasets, engineering parameters, trained weights, or full experiment outputs.
- Some variable names and comments in the original scripts may reflect localized data-column names. Align all column names consistently between training and inference before running experiments.

## Research Scope

This framework supports experiments related to:

- cascade reservoir operation;
- hydropower generation scheduling;
- flood-control and water-supply coordination;
- physically constrained time-series prediction;
- SARIMA-based runoff or inflow forecasting;
- comparison between learning-based prediction, statistical forecasting, and optimization-based scheduling.

## Confidentiality Notice

Due to engineering-parameter confidentiality and data-security considerations, only the core algorithmic components are provided in this repository. For access to complete source code, datasets, and detailed technical documentation, please contact:

```text
zyzhu1128@163.com
```
