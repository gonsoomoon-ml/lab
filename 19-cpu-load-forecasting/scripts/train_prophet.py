#!/usr/bin/env python3
"""
Prophet 모델 학습 및 평가

CPU Load Forecasting을 위한 Prophet time series 모델
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(y_true, y_pred):
    """
    예측 성능 지표 계산

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        dict: MAE, RMSE, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'n_samples': len(y_true)
    }


def plot_predictions(y_true, y_pred, timestamps, output_path, sample_hours=48):
    """
    예측 결과 시각화

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 시간 정보
        output_path: 저장 경로
        sample_hours: 시각화할 시간 (시간)
    """
    sample_minutes = sample_hours * 60
    n_samples = min(sample_minutes, len(y_true))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Plot 1: Time series
    ax1 = axes[0]
    ax1.plot(timestamps[:n_samples], y_true[:n_samples],
             label='Actual', linewidth=2, alpha=0.8, color='#0066CC')
    ax1.plot(timestamps[:n_samples], y_pred[:n_samples],
             label='Prophet Prediction', linewidth=2, alpha=0.8,
             linestyle='--', color='#9370DB')
    ax1.set_ylabel('CPU Load (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Prophet Predictions: First {sample_hours} Hours',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    ax2 = axes[1]
    errors = y_true[:n_samples] - y_pred[:n_samples]
    ax2.plot(timestamps[:n_samples], errors, linewidth=1.5, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (Actual - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Error Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 예측 결과 그래프 저장: {output_path}")


def compare_with_baseline(prophet_metrics, baseline_mape=1.2455, xgboost_mape=0.9293, rf_mape=0.8913):
    """
    Baseline 및 다른 모델들과 성능 비교

    Args:
        prophet_metrics: Prophet 성능 지표
        baseline_mape: Baseline MAPE
        xgboost_mape: XGBoost MAPE
        rf_mape: Random Forest MAPE
    """
    print("\n" + "=" * 80)
    print("모델 성능 비교")
    print("=" * 80)

    improvement_vs_baseline = (baseline_mape - prophet_metrics['MAPE']) / baseline_mape * 100
    vs_xgboost = prophet_metrics['MAPE'] - xgboost_mape
    vs_rf = prophet_metrics['MAPE'] - rf_mape

    print(f"\n{'Model':<25} {'MAPE (%)':<15} {'vs Baseline':<20} {'vs Best ML':<15}")
    print("─" * 80)
    print(f"{'Seasonal Naive (Baseline)':<25} {baseline_mape:<15.4f} {'Baseline':<20} {'-':<15}")
    print(f"{'XGBoost':<25} {xgboost_mape:<15.4f} {'+25.39%':<20} {'-':<15}")
    print(f"{'Random Forest':<25} {rf_mape:<15.4f} {'+28.44%':<20} {'Best ML':<15}")
    print(f"{'Prophet':<25} {prophet_metrics['MAPE']:<15.4f} {improvement_vs_baseline:+.2f}%{'':<13} {vs_rf:+.4f}%{'':<8}")

    print("\n" + "─" * 80)

    if improvement_vs_baseline > 20:
        grade = "S (탁월)"
        status = "✓✓ Baseline을 크게 개선!"
    elif improvement_vs_baseline > 12:
        grade = "A (매우 좋음)"
        status = "✓ Baseline보다 우수"
    elif improvement_vs_baseline > 4:
        grade = "B (좋음)"
        status = "✓ Baseline보다 약간 개선"
    elif improvement_vs_baseline > 0:
        grade = "C (보통)"
        status = "⚠ 개선 미미"
    else:
        grade = "F (실패)"
        status = "✗ Baseline보다 나쁨"

    print(f"평가 등급: {grade}")
    print(f"결과: {status}")

    # Random Forest와 비교
    if abs(vs_rf) < 0.05:
        rf_comparison = "≈ Random Forest와 거의 동일"
    elif vs_rf < 0:
        rf_comparison = f"✓ Random Forest보다 {abs(vs_rf):.4f}% 더 나음"
    else:
        rf_comparison = f"⚠ Random Forest보다 {vs_rf:.4f}% 나쁨"

    print(f"Random Forest 대비: {rf_comparison}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Prophet 모델 학습 및 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train-data', type=str,
                       default='data/train.csv',
                       help='학습 데이터 경로')
    parser.add_argument('--test-data', type=str,
                       default='data/test.csv',
                       help='테스트 데이터 경로')
    parser.add_argument('--output-dir', type=str,
                       default='results/prophet',
                       help='결과 저장 디렉토리')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Prophet 모델 학습")
    print("=" * 80)

    # 1. 데이터 로드
    print(f"\n1. 데이터 로딩...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")

    # 2. Prophet 형식으로 변환
    print(f"\n2. Prophet 형식으로 데이터 변환...")

    # Prophet은 'ds' (datetime)와 'y' (target) 컬럼이 필요
    train_prophet = pd.DataFrame({
        'ds': pd.to_datetime(train_df['timestamp']),
        'y': train_df['average_cpu_load']
    })

    test_prophet = pd.DataFrame({
        'ds': pd.to_datetime(test_df['timestamp']),
        'y': test_df['average_cpu_load']
    })

    print(f"   ✓ 변환 완료")

    # 3. 모델 학습
    print(f"\n3. Prophet 모델 학습...")
    print(f"   Hyperparameters:")
    print(f"     - daily_seasonality: True (일일 패턴)")
    print(f"     - weekly_seasonality: True (주간 패턴)")
    print(f"     - yearly_seasonality: False (데이터 부족)")
    print(f"     - changepoint_prior_scale: 0.05 (추세 변화 민감도)")

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,  # 데이터가 28일밖에 없음
        changepoint_prior_scale=0.05,
        seasonality_mode='additive'
    )

    # 학습 (verbose 최소화)
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)

    model.fit(train_prophet)
    print(f"   ✓ 학습 완료")

    # 4. 예측
    print(f"\n4. 예측 수행...")

    # Train 예측
    train_forecast = model.predict(train_prophet)
    y_train_pred = train_forecast['yhat'].values

    # Test 예측
    test_forecast = model.predict(test_prophet)
    y_test_pred = test_forecast['yhat'].values

    print(f"   ✓ 예측 완료")

    # 5. 성능 평가
    print(f"\n5. 성능 평가...")
    train_metrics = calculate_metrics(train_prophet['y'].values, y_train_pred)
    test_metrics = calculate_metrics(test_prophet['y'].values, y_test_pred)

    print("\n" + "=" * 80)
    print("성능 지표")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12}")
    print("─" * 80)
    print(f"{'Train':<15} {train_metrics['MAE']:<12.4f} {train_metrics['RMSE']:<12.4f} {train_metrics['MAPE']:<12.4f}")
    print(f"{'Test':<15} {test_metrics['MAE']:<12.4f} {test_metrics['RMSE']:<12.4f} {test_metrics['MAPE']:<12.4f}")

    # Baseline 및 다른 모델들과 비교
    compare_with_baseline(test_metrics)

    # 6. 예측 결과 시각화
    print(f"\n6. 예측 결과 시각화...")

    timestamps = test_prophet['ds'].values
    prediction_path = output_dir / 'prophet_predictions.png'
    plot_predictions(test_prophet['y'].values, y_test_pred, timestamps, prediction_path)

    # 7. 모델 저장
    print(f"\n7. 모델 저장...")
    model_path = output_dir / 'prophet_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ 모델 저장: {model_path}")

    # 8. 결과 저장
    print(f"\n8. 결과 저장...")

    # 예측값 저장
    predictions_df = pd.DataFrame({
        'actual': test_prophet['y'].values,
        'predicted': y_test_pred,
        'error': test_prophet['y'].values - y_test_pred
    })
    predictions_path = output_dir / 'prophet_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   ✓ 예측값 저장: {predictions_path}")

    # 메트릭 저장
    metrics_path = output_dir / 'prophet_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("Prophet Model Metrics\n")
        f.write("=" * 40 + "\n\n")
        f.write("[Train Set]\n")
        f.write(f"MAE:  {train_metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {train_metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {train_metrics['MAPE']:.4f}%\n\n")
        f.write("[Test Set]\n")
        f.write(f"MAE:  {test_metrics['MAE']:.4f}\n")
        f.write(f"RMSE: {test_metrics['RMSE']:.4f}\n")
        f.write(f"MAPE: {test_metrics['MAPE']:.4f}%\n\n")
        f.write("[Hyperparameters]\n")
        f.write(f"daily_seasonality: True\n")
        f.write(f"weekly_seasonality: True\n")
        f.write(f"yearly_seasonality: False\n")
        f.write(f"changepoint_prior_scale: 0.05\n")
        f.write(f"seasonality_mode: additive\n")

    print(f"   ✓ 메트릭 저장: {metrics_path}")

    print("\n" + "=" * 80)
    print("✅ Prophet 모델 학습 완료!")
    print("=" * 80)
    print(f"\n저장된 파일:")
    print(f"  - {model_path}")
    print(f"  - {predictions_path}")
    print(f"  - {metrics_path}")
    print(f"  - {prediction_path}")


if __name__ == "__main__":
    main()
