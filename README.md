# An Optimization Learning Simulation Framework for Cascade Reservoir Scheduling

## Project Overview

This project presents a comprehensive optimization learning simulation framework for cascade reservoir scheduling, integrating advanced machine learning techniques with physical constraints to achieve optimal water resource management. The framework combines Long Short-Term Memory (LSTM) neural networks with Deep Q-Network (DQN) reinforcement learning for dynamic weight adjustment, alongside traditional time series forecasting methods such as SARIMA models.

## Key Features

- **Hybrid Learning Architecture**: Integration of LSTM neural networks with DQN for adaptive loss function weighting
- **Physical Constraint Integration**: Incorporation of reservoir operation constraints including water balance, discharge boundaries, and operational cycles
- **Multi-Objective Optimization**: Simultaneous optimization of power generation, flood control, and environmental objectives
- **Time Series Forecasting**: Implementation of SARIMA models for comparative analysis
- **Real-time Inference**: Efficient prediction API for operational deployment

## Technical Architecture

### Core Components

1. **LSTM Prediction Model** (`LSTMPredictor`)
   - Multi-layer LSTM architecture with dropout regularization
   - Sequence-to-sequence prediction for reservoir outflow forecasting
   - Adaptive hidden state management

2. **DQN Agent** (`DQNAgent`)
   - Dynamic weight adjustment for loss function components
   - Experience replay mechanism for stable learning
   - ε-greedy exploration strategy with decay

3. **Physical Constraint Module**
   - Water balance constraints
   - Reservoir release boundary constraints
   - Operational cycle constraints
   - Reservoir Capacity Boundary Constraints
   - Non-negativity constraints

4. **Multi-Objective Optimization**
   - Power generation maximization
   - Flood control optimization
   - Environmental impact minimization

## File Structure

```
├── wddlearning.py          # Main training module with LSTM-DQN framework
├── wddinference_test.py    # Inference API for model deployment
├── runwdd.py              # Example usage script
├── reservoir.m            # MATLAB reservoir optimization model
├── SARIMA/
│   ├── Fun_SARIMA_Forecast.m    # SARIMA forecasting function
│   ├── SARMA_Order_Select.m     # Automatic parameter selection
│   └── creatSARIMA.m           # SARIMA model creation
└── README.md              # Project documentation
```

## Installation and Dependencies

### Python Dependencies
```bash
pip install torch numpy pandas matplotlib scikit-learn joblib
```

### MATLAB Requirements
- MATLAB R2016b or later
- Econometrics Toolbox
- Statistics and Machine Learning Toolbox

## Usage

### 1. Model Training

```python
from wddlearning import main

# Execute training with default parameters
main()
```

### 2. Model Inference

```python
from wddinference_test import ModelAPI
import pandas as pd

# Initialize API
api = ModelAPI()

# Load your data
df = pd.read_excel("your_data.xlsx")

# Single-step prediction
next_prediction = api.predict_next_step(df)

# Multi-step rolling prediction
future_predictions = api.rolling_predict(df, steps=12)
```

### 3. SARIMA Forecasting (MATLAB)

```matlab
% Load data and perform SARIMA forecasting
data = readtable('your_data.xlsx');
[forecast, lower, upper] = Fun_SARIMA_Forecast(data, 12, 3, 3, 2, 2, 12, 'on');
```

## Model Configuration

### LSTM Parameters
- **Input Size**: Number of input features (default: 2)
- **Hidden Size**: LSTM hidden dimension (default: 128)
- **Number of Layers**: LSTM depth (default: 2)
- **Dropout Rate**: Regularization parameter (default: 0.2)
- **Sequence Length**: Historical window size (default: 3)

### DQN Parameters
- **State Dimension**: 5 (episode progress, losses, selected weight)
- **Action Dimension**: 11 (weight values from 0.0 to 1.0)
- **Learning Rate**: 1e-3
- **Gamma**: 0.99 (discount factor)
- **Epsilon Decay**: 0.995

### Physical Constraints
- **Minimum Discharge**: [900, 1260, 1200, 1200] m³/s
- **Maximum Discharge**: [35800, 38800, 40888, 41200] m³/s
- **Water Level Bounds**: Reservoir-specific operational ranges
- **Power Generation Capacity**: [10200, 16000, 12600, 6000] MW

## Performance Metrics

The framework evaluates performance using multiple metrics:

- **Prediction Accuracy**: MAE, MSE, RMSE, R²
- **Hydrological Metrics**: NSE (Nash-Sutcliffe Efficiency), KGE (Kling-Gupta Efficiency)
- **Water Balance Index**: WBI for conservation assessment
- **Physical Constraint Satisfaction**: Violation penalties

## Research Applications

1. **Reservoir Operation Optimization**
2. **Flood Control Management**
3. **Hydropower Generation Planning**
4. **Environmental Impact Assessment**
5. **Climate Change Adaptation Studies**

## Technical Innovations

- **Adaptive Loss Weighting**: DQN-based dynamic adjustment of LSTM and physical loss components
- **Multi-Scale Temporal Modeling**: Integration of short-term LSTM predictions with long-term SARIMA forecasts
- **Constraint-Aware Learning**: Physics-informed neural network architecture
- **Real-time Deployment**: Optimized inference pipeline for operational systems

## Future Enhancements

- Integration of weather forecasting data
- Uncertainty quantification methods
- Advanced reinforcement learning algorithms
- Real-time adaptive control systems


## Confidentiality Notice

**Due to engineering parameter confidentiality and data security considerations, only core algorithmic components are provided in this repository. For access to complete source code, datasets, and detailed technical documentation, please contact: zyzhu1128@163.com**