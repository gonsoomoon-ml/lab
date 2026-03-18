# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UPF CPU Load Forecasting System - A time series forecasting project for predicting 5G core network UPF (User Plane Function) CPU load based on day-of-week, time-of-day, and historical load patterns. The system uses 1-minute granularity data to predict the next 1-minute average load.

**Business Context**: This is for capacity optimization, failure prevention, and enabling predictive auto-scaling of UPF instances in 5G networks.

**Documentation**: Korean-language documentation is available in the `docs/` directory:
- `docs/business-requirements.md` - Business requirements (Korean)
- `docs/research-ko.md` - Research methodology and experimental plan (Korean)
- `docs/data-summary-ko.md` - Dataset analysis and statistics (Korean)
- `README-ko.md` - Project overview and quick start guide (Korean)

## Communication Preferences

**Language**: The user will ask questions in English, but **all responses should be in Korean (한국어)**.
- Answer in Korean language
- Technical terms can use English in parentheses when needed (e.g., "학습 데이터 (training data)")
- Code, commands, and file paths remain in English
- Be direct and concise

## Development Philosophy

**Python Scripts First**: This project uses `.py` files as the primary development approach for rapid iteration, not Jupyter notebooks. Scripts are easier to version control, test, debug, and integrate into pipelines.

**Notebooks for Visualization Only**: Use notebooks sparingly, only for final visualizations, presentations, or interactive exploration of results. All core logic (data processing, feature engineering, model training) lives in `.py` files.

## Project Structure

This is an early-stage project. The expected structure follows data science best practices with script-based development:

```
19-cpu-load-forecasting/
├── data/
│   ├── generate_data.py  # Synthetic data generator script
│   ├── processed/        # Preprocessed datasets with engineered features
│   └── *.csv             # Generated or raw CSV data files
├── src/
│   ├── data/
│   │   ├── loader.py           # Data loading utilities
│   │   └── preprocessing.py    # Feature engineering, missing value handling
│   ├── features/
│   │   └── engineering.py      # Time-based feature extraction (cyclical encoding)
│   ├── models/
│   │   ├── baseline.py         # Naive forecasting models
│   │   ├── statistical.py      # Prophet, ARIMA
│   │   ├── ml.py               # XGBoost, Random Forest
│   │   └── dl.py               # LSTM, Transformer models
│   ├── evaluation/
│   │   └── metrics.py          # MAE, RMSE, MAPE calculations
│   └── visualization/
│       └── plots.py            # Load pattern heatmaps, forecast comparisons
├── scripts/
│   ├── generate_synthetic_data.py  # Create synthetic UPF data
│   ├── run_eda.py                  # Exploratory data analysis
│   ├── train_baseline.py           # Train baseline models
│   ├── train_ml.py                 # Train ML models
│   └── evaluate.py                 # Run evaluation pipeline
├── notebooks/                      # Optional: For presentations/final visualizations only
│   └── results_visualization.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── configs/                    # Model configurations (YAML)
│   ├── baseline.yaml
│   ├── xgboost.yaml
│   └── lstm.yaml
├── models/                     # Saved model artifacts
├── results/                    # Evaluation reports, performance comparisons
├── setup/                      # Environment setup scripts
│   ├── pyproject.toml          # UV dependency management
│   ├── 00_install_uv.sh        # Install UV package manager
│   ├── 01_setup_environment.sh # Create venv and install packages
│   ├── 02_test_environment.sh  # Test environment and generate data
│   └── run_all_setup.sh        # One-click complete setup
├── docs/
│   ├── business-requirements.md    # Business requirements (Korean)
│   ├── research-ko.md              # Research methodology (Korean)
│   ├── data-summary-ko.md          # Dataset analysis (Korean)
│   ├── research.md                 # Research methodology (English)
│   └── data-summary.md             # Dataset analysis (English)
├── requirements.txt            # Pip dependencies (legacy)
├── README.md
└── README-ko.md                # Project README (Korean)
```

## Data Schema

**Input Data** (generated synthetic data or real UPF data, collected every 1 minute):
- `timestamp` (datetime): Data collection timestamp
- `average_cpu_load` (float): 1-minute average CPU load (0-100%) - **PRIMARY PREDICTION TARGET**
- `peak_cpu_load` (float): 1-minute peak CPU load (0-100%), typically 1-3% higher than average
- `active_session_count` (int): Active session count (1,000 - 250,000), proportional to CPU load

**Synthetic Data Pattern** (from `data/generate_data.py`):
- Daily pattern: 4AM low (20%), 9AM peak (70%), 7PM highest (80%)
- Uses PCHIP interpolation for smooth curves with Gaussian noise
- Session count scales linearly with CPU load

**Engineered Features** (extracted from `timestamp`):
- `dayOfWeek` (0-6): Monday=0, Sunday=6
- `hour` (0-23): Hour of day
- `minute` (0-59): Minute within hour
- Cyclical encodings: `hour_sin`, `hour_cos`, `dayOfWeek_sin`, `dayOfWeek_cos`

**Prediction Task**: Given current time and recent `average_cpu_load` values, predict the **next 1-minute average_cpu_load**.

## Development Commands

### Environment Setup (using UV - recommended)
```bash
# One-click setup (installs UV, creates venv, installs dependencies)
cd /home/ubuntu/lab/19-cpu-load-forecasting/setup
./run_all_setup.sh

# Or step-by-step:
./00_install_uv.sh        # Install UV package manager
./01_setup_environment.sh # Create venv and install packages
./02_test_environment.sh  # Test environment and generate data

# Activate environment
cd /home/ubuntu/lab/19-cpu-load-forecasting
source .venv/bin/activate
```

### Manual Setup (alternative)
```bash
# Using pip
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Scripts (Primary Development Workflow)
```bash
# Generate synthetic training data
cd data
python3 generate_data.py  # Creates ai_training_dataset.csv (4 days by default)

# Run exploratory data analysis
python scripts/run_eda.py --input data/ai_training_dataset.csv --output results/eda_report.html

# Preprocess data and engineer features
python -m src.data.preprocessing --input data/ai_training_dataset.csv --output data/processed/

# Train baseline models
python scripts/train_baseline.py --config configs/baseline.yaml

# Train ML models
python scripts/train_ml.py --model xgboost --config configs/xgboost.yaml

# Evaluate model
python scripts/evaluate.py --model-path models/xgboost_best.pkl --test-data data/processed/test.csv
```

### Running Individual Modules
```bash
# Run as module for imports in other scripts
python -m src.data.preprocessing
python -m src.features.engineering
python -m src.models.ml
```

### Notebooks (Optional - Visualization Only)
```bash
# Only if you need to create presentation visualizations
jupyter lab notebooks/results_visualization.ipynb
```

### Testing (once test suite is created)
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=src tests/
```

## Technology Stack

- **Python**: 3.10+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, plotly, seaborn
- **ML**: scikit-learn, xgboost
- **DL**: PyTorch (preferred) or TensorFlow
- **Time Series**: prophet, statsmodels (ARIMA, SARIMA)
- **Experiment Tracking**: MLflow (optional but recommended)

## Key Performance Requirements

- **Prediction Accuracy**: MAPE < 10% (target)
- **Training Time**: < 1 hour (single GPU)
- **Inference Time**: < 1 second per prediction
- **Reproducibility**: Always set random seeds (numpy, sklearn, torch)

## Feature Engineering Guidelines

### Cyclical Feature Encoding
Time-based features (hour, minute, dayOfWeek) have cyclical nature. Use sine/cosine transformation:

```python
# Hour encoding (0-23)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week encoding (0-6)
df['dayOfWeek_sin'] = np.sin(2 * np.pi * df['dayOfWeek'] / 7)
df['dayOfWeek_cos'] = np.cos(2 * np.pi * df['dayOfWeek'] / 7)
```

### Lag Features
Consider adding lag features for temporal dependencies:
- `average_cpu_load_lag1` (t-1 minute)
- `average_cpu_load_lag5` (t-5 minutes)
- `average_cpu_load_lag60` (t-1 hour)
- Rolling statistics: `average_cpu_load_rolling_mean_15min`, `average_cpu_load_rolling_std_15min`

## Model Development Workflow

1. **Baseline**: Start with naive forecast (previous value, moving average)
2. **Statistical**: Try Prophet and ARIMA/SARIMA
3. **ML**: Test XGBoost, Random Forest, Gradient Boosting
4. **DL**: Experiment with LSTM, GRU, Transformer if needed
5. **Ensemble**: Combine best performers

## Evaluation Strategy

- **Metrics**: MAE, RMSE, MAPE (primary)
- **Baseline Comparison**: Must beat naive forecast by 30% RMSE reduction
- **Time-based Split**: Use chronological train/validation/test split (no random shuffle)
- **Error Analysis**: Break down errors by hour-of-day and day-of-week

## Important Constraints

- **No Future Leakage**: Only use data from t-1 and earlier to predict t
- **1-Minute Granularity**: Do not aggregate to hourly/daily for core predictions
- **Missing Data**: UPF may skip sending data; implement forward-fill or interpolation
- **Outlier Handling**: Packet processing can spike during incidents; don't blindly remove outliers

## Data Handling Notes

**Synthetic Data Generation**:
- Use `data/generate_data.py` to create training/testing datasets
- Default: 4 days (5,760 samples), configurable via `days` parameter
- Generates realistic daily patterns with noise for robustness testing
- Minimum recommended: 2 weeks (14 days = 20,160 samples)

## Reproducibility

Always set these seeds at the start of all Python scripts:
```python
import numpy as np
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

## Why Python Scripts Over Notebooks

This project prioritizes `.py` files because:
- **Version Control**: Scripts produce clean, line-by-line diffs in git; notebooks produce noisy JSON diffs
- **Testability**: Easy to write pytest tests for functions in `.py` files
- **Debugging**: IDE debuggers work better with scripts than notebook cells
- **Reproducibility**: Scripts run deterministically from top to bottom; notebooks can have hidden state
- **CI/CD Integration**: Scripts integrate directly into automated pipelines
- **Code Review**: Easier to review clean Python code than notebook JSON

Use notebooks only for final presentation visualizations where interactivity adds value.

## Success Criteria

- [ ] Next 1-minute `average_cpu_load` prediction achieves MAPE < 10%
- [ ] Model beats naive baseline by 30% RMSE reduction
- [ ] Forecast visualization dashboard created
- [ ] Model comparison report completed
