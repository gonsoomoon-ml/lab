#!/usr/bin/env python3
"""
모든 모델 비교 - 개선된 시각화 (v2)

상위 모델과 하위 모델을 분리하여 가독성 향상
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_all_predictions():
    """모든 모델의 예측값 로드"""

    # 1. Baseline (Seasonal Naive)
    baseline_df = pd.read_csv('results/baseline/baseline_seasonal_naive_predictions.csv')
    baseline_pred = baseline_df['seasonal_naive_pred'].values
    baseline_actual = baseline_df['average_cpu_load'].values

    # 2. XGBoost
    xgb_df = pd.read_csv('results/xgboost/xgboost_predictions.csv')
    xgb_pred = xgb_df['predicted'].values
    xgb_actual = xgb_df['actual'].values

    # 3. Random Forest
    rf_df = pd.read_csv('results/random_forest/random_forest_predictions.csv')
    rf_pred = rf_df['predicted'].values
    rf_actual = rf_df['actual'].values

    # 4. Prophet
    prophet_df = pd.read_csv('results/prophet/prophet_predictions.csv')
    prophet_pred = prophet_df['predicted'].values
    prophet_actual = prophet_df['actual'].values

    # 5. LSTM
    lstm_df = pd.read_csv('results/lstm/lstm_predictions.csv')
    lstm_pred = lstm_df['predicted'].values
    lstm_actual = lstm_df['actual'].values

    # Timestamps
    test_original = pd.read_csv('data/test.csv')
    timestamps = pd.to_datetime(test_original['timestamp'])

    return {
        'baseline': (baseline_pred, baseline_actual, timestamps),
        'xgboost': (xgb_pred, xgb_actual, timestamps.iloc[1440:1440+len(xgb_pred)]),
        'random_forest': (rf_pred, rf_actual, timestamps.iloc[1440:1440+len(rf_pred)]),
        'prophet': (prophet_pred, prophet_actual, timestamps),
        'lstm': (lstm_pred, lstm_actual, timestamps.iloc[1500:1500+len(lstm_pred)])
    }


def plot_improved_comparison(predictions_dict, output_path):
    """
    개선된 비교 시각화 - 상위/하위 모델 분리

    Args:
        predictions_dict: 모든 모델의 예측값 딕셔너리
        output_path: 저장 경로
    """

    # XGBoost 기준 시간 범위
    xgb_pred, xgb_actual, xgb_time = predictions_dict['xgboost']

    # 19:00-20:00 구간
    start_idx = 1080
    end_idx = 1140

    if len(xgb_actual) < end_idx:
        print("⚠ 데이터가 부족하여 첫 1시간(0-60분)을 사용합니다.")
        start_idx = 0
        end_idx = 60

    # 데이터 추출
    detail_time = xgb_time.iloc[start_idx:end_idx].reset_index(drop=True)
    detail_actual = xgb_actual[start_idx:end_idx]
    detail_rf = predictions_dict['random_forest'][0][start_idx:end_idx]
    detail_xgb = xgb_pred[start_idx:end_idx]
    detail_lstm = predictions_dict['lstm'][0][start_idx:end_idx]
    detail_baseline = predictions_dict['baseline'][0][start_idx:end_idx]
    detail_prophet = predictions_dict['prophet'][0][start_idx:end_idx]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.5], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    hour_start = detail_time.iloc[0].hour
    hour_end = detail_time.iloc[-1].hour

    # ========== Plot 1: Top Models (RF, XGBoost, LSTM) ==========
    ax1.plot(detail_time, detail_actual,
             label='Actual', linewidth=4, alpha=1.0,
             color='#000000', zorder=10)

    ax1.plot(detail_time, detail_rf,
             label='Random Forest (0.89% MAPE) - Best Overall',
             linewidth=3, alpha=0.9, linestyle='-', color='#228B22')

    ax1.plot(detail_time, detail_xgb,
             label='XGBoost (0.93% MAPE) - 2nd Best',
             linewidth=3, alpha=0.9, linestyle='-', color='#FF8C00')

    ax1.plot(detail_time, detail_lstm,
             label='LSTM (0.96% MAPE) - Deep Learning',
             linewidth=3, alpha=0.9, linestyle='-', color='#DC143C')

    ax1.set_ylabel('CPU Load (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Top 3 ML Models vs Actual ({hour_start:02d}:00-{hour_end:02d}:00) | 1-minute granularity',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=13, framealpha=0.95)
    ax1.grid(True, alpha=0.35, linestyle='--', linewidth=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax1.set_ylim([detail_actual.min() - 0.5, detail_actual.max() + 0.5])

    # ========== Plot 2: Baseline Models (Baseline, Prophet) ==========
    ax2.plot(detail_time, detail_actual,
             label='Actual', linewidth=4, alpha=1.0,
             color='#000000', zorder=10)

    ax2.plot(detail_time, detail_baseline,
             label='Baseline - Seasonal Naive (1.25% MAPE)',
             linewidth=3, alpha=0.8, linestyle='--', color='#4169E1')

    ax2.plot(detail_time, detail_prophet,
             label='Prophet (1.90% MAPE) - Failed Model',
             linewidth=3, alpha=0.8, linestyle=':', color='#9370DB')

    ax2.set_ylabel('CPU Load (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Baseline & Statistical Models vs Actual ({hour_start:02d}:00-{hour_end:02d}:00)',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=13, framealpha=0.95)
    ax2.grid(True, alpha=0.35, linestyle='--', linewidth=1)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax2.set_ylim([detail_actual.min() - 0.5, detail_actual.max() + 0.5])

    # ========== Plot 3: Error Comparison ==========
    errors_rf = detail_actual - detail_rf
    errors_xgb = detail_actual - detail_xgb
    errors_lstm = detail_actual - detail_lstm
    errors_baseline = detail_actual - detail_baseline
    errors_prophet = detail_actual - detail_prophet

    # Fill between for top 3 models
    ax3.fill_between(detail_time, 0, errors_rf, alpha=0.3, color='#228B22', label='RF Error')
    ax3.fill_between(detail_time, 0, errors_xgb, alpha=0.3, color='#FF8C00', label='XGB Error')

    # Lines for others
    ax3.plot(detail_time, errors_lstm, linewidth=2.5, alpha=0.8, color='#DC143C', label='LSTM Error')
    ax3.plot(detail_time, errors_baseline, linewidth=2, alpha=0.7, color='#4169E1', linestyle='--', label='Baseline Error')
    ax3.plot(detail_time, errors_prophet, linewidth=2, alpha=0.7, color='#9370DB', linestyle=':', label='Prophet Error')

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2.5)

    ax3.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Error (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Prediction Errors: Actual - Predicted (Positive = Underestimate, Negative = Overestimate)',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=12, ncol=5, framealpha=0.95)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax3.tick_params(axis='both', labelsize=11)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    plt.xticks(rotation=45)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 개선된 비교 그래프 저장: {output_path}")

    # Statistics
    print(f"\n=== 1시간 구간 통계 ({hour_start:02d}:00-{hour_end:02d}:00) ===")
    print(f"\n{'Model':<20} {'MAE (%)':<12} {'Max Error (%)':<15} {'Std Dev (%)':<12} {'Rank'}")
    print("─" * 80)

    models_stats = [
        ('Random Forest', errors_rf, '🥇'),
        ('XGBoost', errors_xgb, '🥈'),
        ('LSTM', errors_lstm, '🥉'),
        ('Baseline', errors_baseline, ''),
        ('Prophet', errors_prophet, '❌')
    ]

    for name, errors, rank in models_stats:
        mae = np.abs(errors).mean()
        max_err = np.abs(errors).max()
        std = errors.std()
        print(f"{name:<20} {mae:<12.4f} {max_err:<15.4f} {std:<12.4f} {rank}")


def main():
    print("=" * 80)
    print("모든 모델 비교 - 1시간 상세 시각화")
    print("=" * 80)

    # Load predictions
    print("\n데이터 로딩...")
    predictions_dict = load_all_predictions()
    print(f"  ✓ 5개 모델 데이터 로드 완료")

    # Create improved visualization
    print("\n개선된 비교 시각화 생성...")
    output_path = 'results/all_models_1hour_comparison.png'
    plot_improved_comparison(predictions_dict, output_path)

    print("\n" + "=" * 80)
    print("✅ 시각화 완료!")
    print("=" * 80)
    print(f"\n저장 위치: {output_path}")
    print("\n개선 사항:")
    print("  ✓ 상위 3개 ML 모델 (RF, XGBoost, LSTM) 별도 그래프")
    print("  ✓ 하위 2개 모델 (Baseline, Prophet) 별도 그래프")
    print("  ✓ 선 굵기 차별화 (Actual 가장 굵게)")
    print("  ✓ 명확한 색상 대비")
    print("  ✓ 에러 그래프에 fill_between으로 가독성 향상")


if __name__ == "__main__":
    main()
