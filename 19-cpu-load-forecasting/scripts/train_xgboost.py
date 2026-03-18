#!/usr/bin/env python3
"""
XGBoost 모델 학습 및 평가

CPU Load Forecasting을 위한 XGBoost regression 모델
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import joblib
from pathlib import Path
from xgboost import XGBRegressor
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


def plot_feature_importance(model, feature_names, output_path):
    """
    Feature importance 시각화

    Args:
        model: 학습된 XGBoost 모델
        feature_names: Feature 이름 리스트
        output_path: 저장 경로
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 상위 12개 features만 표시
    top_n = min(12, len(feature_names))
    top_indices = indices[:top_n]

    plt.figure(figsize=(12, 8))

    # Horizontal bar plot
    plt.barh(range(top_n), importances[top_indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('XGBoost Feature Importance (Top 12)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # 가장 중요한 것이 위에

    # 값 표시
    for i, idx in enumerate(top_indices):
        plt.text(importances[idx], i, f' {importances[idx]:.4f}',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Feature importance 그래프 저장: {output_path}")


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
             label='XGBoost Prediction', linewidth=2, alpha=0.8,
             linestyle='--', color='#FF6600')
    ax1.set_ylabel('CPU Load (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'XGBoost Predictions: First {sample_hours} Hours',
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


def compare_with_baseline(xgb_metrics, baseline_mape=1.2455):
    """
    Baseline과 성능 비교

    Args:
        xgb_metrics: XGBoost 성능 지표
        baseline_mape: Baseline MAPE
    """
    print("\n" + "=" * 80)
    print("BASELINE 대비 성능 비교")
    print("=" * 80)

    improvement = (baseline_mape - xgb_metrics['MAPE']) / baseline_mape * 100

    print(f"\n{'Model':<25} {'MAPE (%)':<15} {'Improvement':<15}")
    print("─" * 80)
    print(f"{'Seasonal Naive (Baseline)':<25} {baseline_mape:<15.4f} {'Baseline':<15}")
    print(f"{'XGBoost':<25} {xgb_metrics['MAPE']:<15.4f} {improvement:+.2f}%")

    print("\n" + "─" * 80)

    if improvement > 20:
        grade = "S (탁월)"
        status = "✓✓ Baseline을 크게 개선!"
    elif improvement > 12:
        grade = "A (매우 좋음)"
        status = "✓ Baseline보다 우수"
    elif improvement > 4:
        grade = "B (좋음)"
        status = "✓ Baseline보다 약간 개선"
    elif improvement > 0:
        grade = "C (보통)"
        status = "⚠ 개선 미미"
    else:
        grade = "F (실패)"
        status = "✗ Baseline보다 나쁨"

    print(f"평가 등급: {grade}")
    print(f"결과: {status}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost 모델 학습 및 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train-data', type=str,
                       default='data/processed/train_processed.csv',
                       help='학습 데이터 경로')
    parser.add_argument('--test-data', type=str,
                       default='data/processed/test_processed.csv',
                       help='테스트 데이터 경로')
    parser.add_argument('--output-dir', type=str,
                       default='results/xgboost',
                       help='결과 저장 디렉토리')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='XGBoost estimators 개수')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Tree 최대 깊이')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("XGBoost 모델 학습")
    print("=" * 80)

    # 1. 데이터 로드
    print(f"\n1. 데이터 로딩...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    print(f"   Train: {len(train_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")

    # Feature와 Target 분리
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    feature_names = X_train.columns.tolist()
    print(f"   Features: {len(feature_names)}개")

    # 2. 모델 학습
    print(f"\n2. XGBoost 모델 학습...")
    print(f"   Hyperparameters:")
    print(f"     - n_estimators: {args.n_estimators}")
    print(f"     - max_depth: {args.max_depth}")
    print(f"     - learning_rate: {args.learning_rate}")

    model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(X_train, y_train)
    print(f"   ✓ 학습 완료")

    # 3. 예측
    print(f"\n3. 예측 수행...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"   ✓ 예측 완료")

    # 4. 성능 평가
    print(f"\n4. 성능 평가...")
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    print("\n" + "=" * 80)
    print("성능 지표")
    print("=" * 80)

    print(f"\n{'Dataset':<15} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12}")
    print("─" * 80)
    print(f"{'Train':<15} {train_metrics['MAE']:<12.4f} {train_metrics['RMSE']:<12.4f} {train_metrics['MAPE']:<12.4f}")
    print(f"{'Test':<15} {test_metrics['MAE']:<12.4f} {test_metrics['RMSE']:<12.4f} {test_metrics['MAPE']:<12.4f}")

    # Baseline과 비교
    compare_with_baseline(test_metrics)

    # 5. Feature Importance 분석
    print(f"\n5. Feature Importance 분석...")

    importance_path = output_dir / 'xgboost_feature_importance.png'
    plot_feature_importance(model, feature_names, importance_path)

    # Feature importance 출력
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n   Top 5 Important Features:")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"     {i+1}. {feature_names[idx]:<40} {importances[idx]:.4f}")

    # 6. 예측 결과 시각화
    print(f"\n6. 예측 결과 시각화...")

    # Test 데이터에 timestamp가 없으므로 원본에서 로드
    test_original = pd.read_csv('data/test.csv')
    timestamps = pd.to_datetime(test_original['timestamp'])

    # Feature engineering으로 1440개가 제거되었으므로 그만큼 건너뛰기
    timestamps_aligned = timestamps.iloc[1440:1440+len(y_test)].values

    prediction_path = output_dir / 'xgboost_predictions.png'
    plot_predictions(y_test.values, y_test_pred, timestamps_aligned, prediction_path)

    # 7. 모델 저장
    print(f"\n7. 모델 저장...")
    model_path = output_dir / 'xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"   ✓ 모델 저장: {model_path}")

    # 8. 결과 저장
    print(f"\n8. 결과 저장...")

    # 예측값 저장
    predictions_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_test_pred,
        'error': y_test.values - y_test_pred
    })
    predictions_path = output_dir / 'xgboost_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   ✓ 예측값 저장: {predictions_path}")

    # 메트릭 저장
    metrics_path = output_dir / 'xgboost_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("XGBoost Model Metrics\n")
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
        f.write(f"n_estimators: {args.n_estimators}\n")
        f.write(f"max_depth: {args.max_depth}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")

    print(f"   ✓ 메트릭 저장: {metrics_path}")

    print("\n" + "=" * 80)
    print("✅ XGBoost 모델 학습 완료!")
    print("=" * 80)
    print(f"\n저장된 파일:")
    print(f"  - {model_path}")
    print(f"  - {predictions_path}")
    print(f"  - {metrics_path}")
    print(f"  - {importance_path}")
    print(f"  - {prediction_path}")


if __name__ == "__main__":
    main()
