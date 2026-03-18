#!/usr/bin/env python3
"""
Seasonal Naive Baseline for CPU Load Forecasting

Prediction Strategy: pred(t) = actual(t-1440)
Uses the CPU load from the same time yesterday as the prediction.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, mask=None):
    """
    Calculate forecasting metrics: MAE, RMSE, MAPE

    Args:
        y_true: Actual values
        y_pred: Predicted values
        mask: Boolean mask for valid predictions (optional)

    Returns:
        dict: Dictionary of metrics
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # Remove any remaining NaN values
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'n_samples': len(y_true)
    }


def seasonal_naive_forecast(train_df, test_df, target_col='average_cpu_load', lag=1440):
    """
    Generate seasonal naive forecasts.

    Args:
        train_df: Training dataframe with timestamp and target columns
        test_df: Test dataframe with timestamp and target columns
        target_col: Name of the target column
        lag: Seasonal lag (1440 = 1 day for 1-minute data)

    Returns:
        tuple: (test_df with predictions, metrics)
    """
    print(f"🔮 Generating Seasonal Naive Forecasts (lag={lag} minutes)")
    print("─" * 80)

    # Combine train and test for seamless lookback
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Ensure timestamp is datetime
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    # Combine datasets
    combined = pd.concat([train_df, test_df], ignore_index=True).sort_values('timestamp')

    # Generate seasonal naive prediction: value from 'lag' minutes ago
    combined['seasonal_naive_pred'] = combined[target_col].shift(lag)

    # Extract test predictions
    test_start = test_df['timestamp'].min()
    test_end = test_df['timestamp'].max()

    test_predictions = combined[
        (combined['timestamp'] >= test_start) &
        (combined['timestamp'] <= test_end)
    ].copy()

    # Calculate metrics
    valid_mask = ~test_predictions['seasonal_naive_pred'].isna()
    metrics = calculate_metrics(
        test_predictions[target_col].values,
        test_predictions['seasonal_naive_pred'].values,
        mask=valid_mask
    )

    print(f"✓ Generated {metrics['n_samples']:,} predictions")
    print(f"  (Note: First {lag} test samples have no prediction due to lookback requirement)")

    return test_predictions, metrics


def print_results(metrics):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 80)
    print("📊 BASELINE EVALUATION RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Value':<15} {'Status':<20}")
    print("─" * 80)

    mae_status = "✓ Good" if metrics['MAE'] < 5 else "⚠ Check"
    rmse_status = "✓ Good" if metrics['RMSE'] < 6 else "⚠ Check"
    mape_status = "✓✓ Excellent" if metrics['MAPE'] < 10 else "⚠ Needs improvement"

    print(f"{'MAE':<20} {metrics['MAE']:<15.4f} {mae_status:<20}")
    print(f"{'RMSE':<20} {metrics['RMSE']:<15.4f} {rmse_status:<20}")
    print(f"{'MAPE':<20} {metrics['MAPE']:<15.4f}% {mape_status:<20}")
    print(f"{'Samples Evaluated':<20} {metrics['n_samples']:<15,}")

    print("\n" + "─" * 80)
    print("🎯 Target: MAPE < 10%")
    if metrics['MAPE'] < 10:
        print(f"✓ BASELINE PASSES (MAPE = {metrics['MAPE']:.2f}%)")
        print(f"  ML models must beat {metrics['MAPE']:.2f}% MAPE to be useful")
    else:
        print(f"⚠ BASELINE FAILS (MAPE = {metrics['MAPE']:.2f}%)")
        print(f"  This suggests the data may be more challenging than expected")
    print("=" * 80)


def plot_predictions(test_predictions, output_path=None, sample_hours=48):
    """
    Plot actual vs predicted values for visual inspection.

    Args:
        test_predictions: DataFrame with timestamp, actual, and predicted values
        output_path: Path to save the plot (optional)
        sample_hours: Number of hours to plot (default: 48)
    """
    # Plot first N hours for visibility
    sample_minutes = sample_hours * 60
    plot_df = test_predictions.head(sample_minutes).copy()

    if len(plot_df) == 0:
        print("⚠ No data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Time series comparison
    ax1.plot(plot_df['timestamp'], plot_df['average_cpu_load'],
             label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(plot_df['timestamp'], plot_df['seasonal_naive_pred'],
             label='Seasonal Naive Prediction', linewidth=2, alpha=0.8, linestyle='--')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU Load (%)')
    ax1.set_title(f'Seasonal Naive Baseline: First {sample_hours} Hours of Test Set')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error distribution
    errors = plot_df['average_cpu_load'] - plot_df['seasonal_naive_pred']
    errors = errors.dropna()

    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Seasonal Naive Baseline for CPU Load Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python baseline_seasonal_naive.py --train-data data/train.csv --test-data data/test.csv
        """
    )

    parser.add_argument(
        '--train-data',
        type=str,
        default='data/train.csv',
        help='Path to training data CSV'
    )

    parser.add_argument(
        '--test-data',
        type=str,
        default='data/test.csv',
        help='Path to test data CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline',
        help='Directory to save results (default: results/baseline/)'
    )

    parser.add_argument(
        '--lag',
        type=int,
        default=1440,
        help='Seasonal lag in minutes (default: 1440 = 1 day)'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SEASONAL NAIVE BASELINE EVALUATION")
    print("=" * 80)

    # Load data
    print(f"\n📂 Loading data...")
    print(f"  Train: {args.train_data}")
    print(f"  Test:  {args.test_data}")

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"\n✓ Data loaded")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples:  {len(test_df):,}")

    # Generate forecasts
    test_predictions, metrics = seasonal_naive_forecast(
        train_df, test_df, lag=args.lag
    )

    # Print results
    print_results(metrics)

    # Save predictions
    output_file = output_dir / 'baseline_seasonal_naive_predictions.csv'
    test_predictions.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to: {output_file}")

    # Save metrics
    metrics_file = output_dir / 'baseline_seasonal_naive_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("Seasonal Naive Baseline Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"MAE:  {metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {metrics['MAPE']:.4f}%\n")
        f.write(f"\nSamples: {metrics['n_samples']:,}\n")
        f.write(f"Lag: {args.lag} minutes\n")

    print(f"✓ Metrics saved to: {metrics_file}")

    # Generate plots if requested
    if args.plot:
        print("\n📊 Generating visualization...")
        plot_file = output_dir / 'baseline_seasonal_naive_plot.png'
        plot_predictions(test_predictions, output_path=plot_file)

    print("\n" + "=" * 80)
    print("✅ BASELINE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
