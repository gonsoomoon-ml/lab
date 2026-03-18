#!/usr/bin/env python3
"""
LSTM 결과 상세 시각화

Baseline과 동일한 스타일로 1시간 상세 뷰 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_1hour_detailed_comparison(actual, predicted, timestamps, output_path):
    """
    1시간 상세 비교 시각화 (Baseline 스타일)

    Args:
        actual: 실제값
        predicted: 예측값
        timestamps: 시간 정보
        output_path: 저장 경로
    """
    # 19:00-20:00 구간 선택 (1080-1140번째 인덱스)
    start_idx = 1080
    end_idx = 1140

    if len(actual) < end_idx:
        print("⚠ 데이터가 부족하여 첫 1시간(0-60분)을 사용합니다.")
        start_idx = 0
        end_idx = 60

    detail_actual = actual[start_idx:end_idx]
    detail_pred = predicted[start_idx:end_idx]
    detail_time = timestamps[start_idx:end_idx]

    # Calculate errors
    errors = detail_actual - detail_pred

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

    # Top plot: Actual vs Predicted
    ax1 = axes[0]

    # Plot with markers
    ax1.plot(detail_time, detail_actual,
             label='Actual (실제값)', linewidth=2.5, alpha=0.9,
             color='#0066CC', marker='o', markersize=4,
             markeredgecolor='white', markeredgewidth=0.5)

    ax1.plot(detail_time, detail_pred,
             label='LSTM Prediction (예측값)', linewidth=2, alpha=0.85,
             linestyle='--', color='#DC143C', marker='s', markersize=4,
             markeredgecolor='white', markeredgewidth=0.5)

    # Add value annotations (every 15 minutes)
    for i in [0, 15, 30, 45, 59]:
        if i < len(detail_actual):
            # Actual value (blue)
            ax1.annotate(f'{detail_actual.iloc[i]:.1f}',
                        xy=(detail_time.iloc[i], detail_actual.iloc[i]),
                        xytext=(0, 10), textcoords='offset points',
                        fontsize=9, color='#0066CC', fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', edgecolor='#0066CC', alpha=0.8))

            # Predicted value (red)
            ax1.annotate(f'{detail_pred.iloc[i]:.1f}',
                        xy=(detail_time.iloc[i], detail_pred.iloc[i]),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=9, color='#DC143C', fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', edgecolor='#DC143C', alpha=0.8))

    # Format time axis
    hour_start = detail_time.iloc[0].hour
    hour_end = detail_time.iloc[-1].hour

    ax1.set_ylabel('CPU Load (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'1-Hour Detailed View: Evening Peak Period ({hour_start:02d}:00 - {hour_end:02d}:00)\nEach point = 1 minute',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle=':', linewidth=1.2)
    ax1.tick_params(axis='both', labelsize=11)

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    # Bottom plot: Error bars
    ax2 = axes[1]
    ax2.bar(detail_time, errors, width=0.0005, color='red', alpha=0.7,
            label='Error = Actual - Predicted')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Prediction Error Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()

    print(f"✓ 1시간 상세 뷰 저장: {output_path}")

    # Print statistics
    print(f"\n1시간 구간 통계 ({hour_start:02d}:00-{hour_end:02d}:00):")
    print(f"  실제 CPU 범위:  {detail_actual.min():.2f}% - {detail_actual.max():.2f}%")
    print(f"  예측 CPU 범위:  {detail_pred.min():.2f}% - {detail_pred.max():.2f}%")
    print(f"  평균 절대 오차:  {errors.abs().mean():.4f}%")
    print(f"  최대 오차:       {errors.abs().max():.4f}%")
    print(f"  오차 표준편차:   {errors.std():.4f}%")


def main():
    print("=" * 80)
    print("LSTM 상세 시각화")
    print("=" * 80)

    # Load predictions
    print("\n데이터 로딩...")
    predictions = pd.read_csv('results/lstm/lstm_predictions.csv')

    # Load original test data for timestamps
    test_original = pd.read_csv('data/test.csv')
    timestamps = pd.to_datetime(test_original['timestamp'])

    # LSTM uses sequence_length=60 + lag features (1440)
    # Total offset = 1440 + 60 = 1500
    offset = 1500
    timestamps_aligned = timestamps.iloc[offset:offset+len(predictions)]

    # Reset index for proper alignment
    actual = predictions['actual'].reset_index(drop=True)
    predicted = predictions['predicted'].reset_index(drop=True)
    timestamps_aligned = timestamps_aligned.reset_index(drop=True)

    print(f"  총 {len(predictions):,}개 샘플")

    # Create 1-hour detailed view
    print("\n1시간 상세 시각화 생성...")
    output_path = 'results/lstm/lstm_1hour_detailed.png'
    plot_1hour_detailed_comparison(actual, predicted, timestamps_aligned, output_path)

    print("\n" + "=" * 80)
    print("✅ 시각화 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
