#!/usr/bin/env python3
"""
Feature Engineering for CPU Load Forecasting

Standard Set (12 features):
- Lag features: 1, 5, 15, 60, 120, 1440 minutes
- Rolling statistics: mean(15min, 60min), std(15min)
- Temporal: day_of_week (sin/cos), is_weekend
"""

import pandas as pd
import numpy as np


def create_lag_features(df, target_col='average_cpu_load', lags=[1, 5, 15, 60, 120, 1440]):
    """
    과거 값을 lag feature로 생성

    Args:
        df: DataFrame with timestamp and target column
        target_col: 대상 컬럼명
        lags: lag 분 리스트

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    for lag in lags:
        df[f'{target_col}_lag_{lag}min'] = df[target_col].shift(lag)

    return df


def create_rolling_features(df, target_col='average_cpu_load', windows=[15, 60]):
    """
    이동 통계량 (rolling mean, std) 생성

    Args:
        df: DataFrame with target column
        target_col: 대상 컬럼명
        windows: window size 리스트 (분 단위)

    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}min'] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )

        # Rolling std (15분만)
        if window == 15:
            df[f'{target_col}_rolling_std_{window}min'] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )

    return df


def create_temporal_features(df, timestamp_col='timestamp'):
    """
    시간 관련 feature 생성 (day_of_week만)

    Args:
        df: DataFrame with timestamp column
        timestamp_col: timestamp 컬럼명

    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[timestamp_col].dt.dayofweek

    # Cyclical encoding for day_of_week
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Is weekend (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Drop intermediate column
    df = df.drop(columns=['day_of_week'])

    return df


def engineer_features(df, target_col='average_cpu_load', timestamp_col='timestamp'):
    """
    Standard Set feature engineering 전체 파이프라인

    생성되는 features (12개):
    1. lag_1min, lag_5min, lag_15min, lag_60min, lag_120min, lag_1440min (6개)
    2. rolling_mean_15min, rolling_std_15min, rolling_mean_60min (3개)
    3. day_of_week_sin, day_of_week_cos, is_weekend (3개)

    Args:
        df: 입력 DataFrame (timestamp, average_cpu_load 필수)
        target_col: 예측 대상 컬럼명
        timestamp_col: timestamp 컬럼명

    Returns:
        DataFrame with all engineered features
    """
    print("=" * 80)
    print("Feature Engineering - Standard Set")
    print("=" * 80)

    df = df.copy()

    # 1. Lag features
    print("\n1. Creating lag features...")
    lags = [1, 5, 15, 60, 120, 1440]
    df = create_lag_features(df, target_col=target_col, lags=lags)
    print(f"   ✓ Created {len(lags)} lag features")

    # 2. Rolling statistics
    print("\n2. Creating rolling statistics...")
    df = create_rolling_features(df, target_col=target_col, windows=[15, 60])
    print(f"   ✓ Created 3 rolling features (mean_15min, std_15min, mean_60min)")

    # 3. Temporal features
    print("\n3. Creating temporal features...")
    df = create_temporal_features(df, timestamp_col=timestamp_col)
    print(f"   ✓ Created 3 temporal features (day_of_week_sin, day_of_week_cos, is_weekend)")

    print("\n" + "=" * 80)
    print("Feature Engineering Complete")
    print("=" * 80)

    return df


def get_feature_columns(target_col='average_cpu_load'):
    """
    Standard Set의 feature 컬럼 리스트 반환

    Returns:
        list: feature 컬럼명 리스트
    """
    features = [
        # Lag features (6)
        f'{target_col}_lag_1min',
        f'{target_col}_lag_5min',
        f'{target_col}_lag_15min',
        f'{target_col}_lag_60min',
        f'{target_col}_lag_120min',
        f'{target_col}_lag_1440min',

        # Rolling statistics (3)
        f'{target_col}_rolling_mean_15min',
        f'{target_col}_rolling_std_15min',
        f'{target_col}_rolling_mean_60min',

        # Temporal features (3)
        'day_of_week_sin',
        'day_of_week_cos',
        'is_weekend',
    ]

    return features


def prepare_train_test(train_df, test_df, target_col='average_cpu_load', timestamp_col='timestamp'):
    """
    Train/Test 데이터에 feature engineering 적용 및 전처리

    Args:
        train_df: 학습 데이터
        test_df: 테스트 데이터
        target_col: 예측 대상 컬럼
        timestamp_col: timestamp 컬럼

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_names)
    """
    print("=" * 80)
    print("TRAIN/TEST DATA PREPARATION")
    print("=" * 80)

    # Feature engineering
    print("\n[Train Data]")
    train_engineered = engineer_features(train_df, target_col=target_col, timestamp_col=timestamp_col)

    print("\n[Test Data]")
    test_engineered = engineer_features(test_df, target_col=target_col, timestamp_col=timestamp_col)

    # Get feature columns
    feature_cols = get_feature_columns(target_col=target_col)

    # Remove rows with NaN (due to lag/rolling windows)
    print("\n" + "=" * 80)
    print("Removing NaN rows (due to lag features)...")
    print("=" * 80)

    train_clean = train_engineered.dropna()
    test_clean = test_engineered.dropna()

    print(f"\nTrain: {len(train_df):,} → {len(train_clean):,} samples (removed {len(train_df) - len(train_clean):,})")
    print(f"Test:  {len(test_df):,} → {len(test_clean):,} samples (removed {len(test_df) - len(test_clean):,})")

    # Split X and y
    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]

    X_test = test_clean[feature_cols]
    y_test = test_clean[target_col]

    print("\n" + "=" * 80)
    print("Data Preparation Complete")
    print("=" * 80)
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print("\nFeature list:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")

    return X_train, y_train, X_test, y_test, feature_cols


if __name__ == "__main__":
    # 간단한 테스트
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering for CPU Load Forecasting")
    parser.add_argument('--train-data', type=str, default='data/train.csv', help='Train CSV path')
    parser.add_argument('--test-data', type=str, default='data/test.csv', help='Test CSV path')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')

    args = parser.parse_args()

    # Load data
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"\nOriginal data:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    # Prepare features
    X_train, y_train, X_test, y_test, feature_names = prepare_train_test(train_df, test_df)

    # Save processed data
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Combine X and y for saving
    train_processed = X_train.copy()
    train_processed['target'] = y_train

    test_processed = X_test.copy()
    test_processed['target'] = y_test

    train_processed.to_csv(f'{args.output_dir}/train_processed.csv', index=False)
    test_processed.to_csv(f'{args.output_dir}/test_processed.csv', index=False)

    print(f"\n✓ Processed data saved to {args.output_dir}/")
    print(f"  - train_processed.csv")
    print(f"  - test_processed.csv")
