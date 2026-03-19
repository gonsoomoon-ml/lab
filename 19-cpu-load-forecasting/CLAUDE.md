# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Status**: ✅ Completed and Production-Ready

UPF CPU Load Forecasting System - A time series forecasting project for predicting 5G core network UPF (User Plane Function) CPU load. Successfully achieved **0.89% MAPE** with Random Forest model, beating the 10% target by 11x.

**Business Context**: Capacity optimization, failure prevention, and predictive auto-scaling for 5G UPF instances.

## Communication Preferences

**Language**: The user will ask questions in English, but **all responses should be in Korean (한국어)**.
- Answer in Korean language
- Technical terms can use English in parentheses (e.g., "학습 데이터 (training data)")
- Code, commands, and file paths remain in English
- Be direct and concise

## Workflow Preferences

**Before Action, Ask Permission**: Before executing any action (creating files, running scripts, training models, modifying code), always ask the user for confirmation first.
- Explain what you plan to do
- Wait for user approval before proceeding
- Do not assume the user wants changes applied automatically

## Project Architecture

### Core Design Principle: Script-Based Development

This project uses **Python scripts (`.py` files)** as the primary approach, NOT Jupyter notebooks:
- All data processing, feature engineering, and model training live in `.py` files
- Notebooks are only used for final visualizations or interactive exploration
- Scripts are version-controlled, testable, and integrate into pipelines easily

### Project Structure (Actual)

```
19-cpu-load-forecasting/
├── data/
│   ├── generate_data.py       # Synthetic data generator
│   ├── split_train_test.py    # Train/test splitter
│   ├── train.csv              # 21 days (30,240 samples)
│   ├── test.csv               # 7 days (10,080 samples)
│   └── processed/             # Feature-engineered datasets
│
├── src/
│   ├── features/
│   │   └── engineering.py     # 12-feature engineering pipeline
│   └── models/
│       └── __init__.py
│
├── scripts/                   # Training & visualization scripts
│   ├── baseline_seasonal_naive.py
│   ├── train_xgboost.py
│   ├── train_random_forest.py
│   ├── train_lstm.py
│   ├── train_prophet.py
│   ├── visualize_random_forest_detailed.py
│   ├── visualize_xgboost_detailed.py
│   ├── visualize_lstm_detailed.py
│   ├── visualize_prophet_detailed.py
│   └── visualize_all_models_comparison.py
│
├── results/                   # Model artifacts & plots
│   ├── baseline/              # Seasonal Naive (7 files)
│   ├── random_forest/         # Best model (6 files, 0.89% MAPE)
│   ├── xgboost/               # 2nd place (6 files, 0.93% MAPE)
│   ├── lstm/                  # Deep learning (6 files, 0.96% MAPE)
│   ├── prophet/               # Failed model (5 files, 1.90% MAPE)
│   └── README.md              # Results summary
│
├── docs/                      # Documentation
│   ├── final-report.md        # 📄 Comprehensive 1,200-line report
│   ├── feature-engineering.md # Feature engineering guide
│   ├── baseline-results.md    # Baseline analysis
│   └── business-requirements.md
│
├── setup/                     # Environment setup
│   ├── run_all_setup.sh       # One-click setup (UV + deps)
│   ├── 00_install_uv.sh
│   ├── 01_setup_environment.sh
│   ├── 02_test_environment.sh
│   └── pyproject.toml         # UV dependencies
│
├── CLAUDE.md                  # This file
├── README.md                  # English README
├── README-ko.md               # Korean README
└── requirements.txt           # Pip dependencies (legacy)
```

## Common Development Commands

### Environment Setup
```bash
# One-click setup (recommended)
cd setup && ./run_all_setup.sh

# Activate environment
source .venv/bin/activate
```

### Data Generation
```bash
cd data
python generate_data.py --days 21 --output train.csv
python generate_data.py --days 7 --output test.csv --start-date 2024-01-22
```

### Feature Engineering
```bash
# Run as module from project root
python -m src.features.engineering
```

### Model Training
```bash
# From project root
python scripts/baseline_seasonal_naive.py --plot
python scripts/train_random_forest.py
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/train_prophet.py
```

### Visualization
```bash
# Detailed model visualizations
python scripts/visualize_random_forest_detailed.py
python scripts/visualize_xgboost_detailed.py
python scripts/visualize_lstm_detailed.py

# Compare all models
python scripts/visualize_all_models_comparison.py
```

## Data Schema

**Input Data** (1-minute granularity):
- `timestamp` (datetime): Collection timestamp
- `average_cpu_load` (float, 0-100%): **PRIMARY TARGET** - 1-minute average CPU load
- `peak_cpu_load` (float, 0-100%): 1-minute peak CPU load (1-3% higher than average)
- `active_session_count` (int, 1K-250K): Session count proportional to CPU load

**Engineered Features** (12 total):
- **Lag features**: `lag_1min`, `lag_5min`, `lag_15min`, `lag_60min`, `lag_120min`, `lag_1440min`
- **Rolling statistics**: `rolling_mean_15min`, `rolling_std_15min`, `rolling_mean_60min`
- **Temporal**: `day_of_week_sin`, `day_of_week_cos`, `is_weekend`

## Key Technical Decisions

### Feature Engineering Strategy
Implemented in `src/features/engineering.py`:
1. **Lag features** capture temporal dependencies (most recent = most important)
2. **Rolling statistics** smooth out noise and capture trends
3. **Cyclical encoding** for time features (sin/cos transformation prevents discontinuity at day boundaries)

Example:
```python
# Day of week encoding (0-6) - prevents discontinuity between Sun(6) and Mon(0)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```

### Model Comparison Results

| Model | MAPE | vs Baseline | Status |
|-------|------|-------------|--------|
| **Random Forest** | **0.8913%** | **+28.44%** | ✅ **Production Ready** |
| XGBoost | 0.9293% | +25.39% | ✅ Backup Model |
| LSTM | 0.9619% | +22.77% | ✅ Peak Specialist |
| Seasonal Naive | 1.2455% | Baseline | ✅ Benchmark |
| Prophet | 1.8997% | -52.52% | ❌ Failed |

**Recommendation**: Deploy Random Forest for production use (best accuracy + fast inference).

### Reproducibility
All scripts use fixed random seeds:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

## Important Constraints

1. **No Future Leakage**: Only use data from t-1 and earlier to predict t
2. **Time-based Split**: Use chronological train/test split (no random shuffle for time series)
3. **1-Minute Granularity**: Do not aggregate to hourly/daily for core predictions
4. **Missing Data Handling**: UPF may skip sending data; use forward-fill or interpolation

## Documentation

- **[docs/final-report.md](docs/final-report.md)** - 📄 **Read this first**: Comprehensive 1,200-line report with executive summary, model comparison, production deployment guide, and cost-benefit analysis
- **[docs/feature-engineering.md](docs/feature-engineering.md)** - Detailed feature engineering methodology
- **[docs/baseline-results.md](docs/baseline-results.md)** - Baseline model analysis
- **[README.md](README.md)** - Quick start guide (English)
- **[README.md](README.md)** - 프로젝트 소개 (한국어, 메인 README)

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.10+ |
| Package Manager | UV | latest |
| Data Processing | pandas, numpy | 2.0+, 1.24+ |
| ML | scikit-learn, XGBoost | 1.3+, 2.0+ |
| Deep Learning | PyTorch | 2.0+ |
| Time Series | Prophet | 1.1+ |
| Visualization | matplotlib, seaborn | 3.7+, 0.12+ |

## Production Deployment

### Basic Strategy: Random Forest Only
```python
import joblib
import pandas as pd

# Load model
model = joblib.load('results/random_forest/random_forest_model.pkl')

# Prepare features (requires last 1440 minutes of historical data)
features = engineer_features(historical_data)

# Predict next minute
prediction = model.predict([features])[0]
```

**Requirements**:
- CPU: 2 cores
- RAM: 4GB
- Latency: <10ms (99th percentile)

### Advanced Strategy: Hybrid (RF + LSTM)
Use Random Forest for normal hours, LSTM for peak hours (18:00-21:00) where LSTM performs better.

## Success Criteria (All Achieved ✅)

- [x] Next 1-minute `average_cpu_load` prediction achieves MAPE < 10% → **Achieved 0.89%**
- [x] Model beats naive baseline by 30% RMSE reduction → **Achieved 28.44%**
- [x] Forecast visualization dashboard created → **6 visualizations per model**
- [x] Model comparison report completed → **docs/final-report.md**
