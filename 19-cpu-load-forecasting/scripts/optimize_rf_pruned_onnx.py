#!/usr/bin/env python3
"""
Random Forest Tree Pruning + ONNX 변환

트리 개수와 깊이를 줄여 재학습한 후 ONNX로 변환합니다.
여러 pruning 설정을 비교하여 최적의 크기/성능 밸런스를 찾습니다.

사용법:
    python scripts/optimize_rf_pruned_onnx.py
"""

import time
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxconverter_common import float16
import onnxruntime as ort

warnings.filterwarnings('ignore')

SEED = 42


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def train_pruned_rf(X_train, y_train, n_estimators, max_depth):
    """경량 Random Forest 학습"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=SEED,
        n_jobs=-1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time


def to_onnx_fp16(sklearn_model, n_features, fp32_path, fp16_path):
    """sklearn → ONNX fp32 → fp16 변환"""
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type, target_opset=15)
    onnx.save(onnx_model, str(fp32_path))

    model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))

    return Path(fp32_path).stat().st_size / 1024 / 1024, Path(fp16_path).stat().st_size / 1024 / 1024


def evaluate_onnx(model_path, X_test, y_test):
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    y_pred = sess.run(None, {input_name: X_test.astype(np.float32)})[0].flatten()
    return calculate_metrics(y_test, y_pred), y_pred


def benchmark_latency(model_path, X_sample, n_iter=1000):
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    sample = X_sample[:1].astype(np.float32)
    latencies = []
    for _ in range(n_iter):
        start = time.perf_counter()
        sess.run(None, {input_name: sample})
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p99_ms': np.percentile(latencies, 99),
    }


def main():
    parser = argparse.ArgumentParser(description="RF Tree Pruning + ONNX 변환")
    parser.add_argument('--train-data', type=str, default='data/processed/train_processed.csv')
    parser.add_argument('--test-data', type=str, default='data/processed/test_processed.csv')
    parser.add_argument('--output-dir', type=str, default='results/optimized')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Random Forest Tree Pruning + ONNX 변환")
    print("=" * 90)

    # 데이터 로드
    print("\n[Step 1] 데이터 로드")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)

    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    n_features = X_train.shape[1]

    print(f"  Train: {len(y_train):,}, Test: {len(y_test):,}, Features: {n_features}")

    # 원본 모델 성능 (비교 기준)
    print("\n[Step 2] 원본 모델 성능 로드")
    original_pkl = Path('results/random_forest/random_forest_model.pkl')
    original_size = original_pkl.stat().st_size / 1024 / 1024
    original_model = joblib.load(original_pkl)
    original_metrics = calculate_metrics(y_test, original_model.predict(X_test))
    print(f"  Original: {original_size:.2f}MB, MAPE: {original_metrics['MAPE']:.4f}%")

    # Pruning 설정 비교
    configs = [
        {'n_estimators': 50, 'max_depth': 15},
        {'n_estimators': 30, 'max_depth': 12},
        {'n_estimators': 20, 'max_depth': 10},
        {'n_estimators': 10, 'max_depth': 8},
    ]

    print(f"\n[Step 3] {len(configs)}가지 Pruning 설정 학습 및 ONNX 변환")
    print("-" * 90)

    results = []

    for cfg in configs:
        n_est = cfg['n_estimators']
        depth = cfg['max_depth']
        label = f"RF(n={n_est},d={depth})"
        print(f"\n  {label} 학습 중...")

        # 학습
        model, train_time = train_pruned_rf(X_train, y_train, n_est, depth)

        # sklearn 평가
        sklearn_metrics = calculate_metrics(y_test, model.predict(X_test))

        # pkl 저장 (크기 비교용)
        pkl_path = output_dir / f'rf_pruned_n{n_est}_d{depth}.pkl'
        joblib.dump(model, pkl_path)
        pkl_size = pkl_path.stat().st_size / 1024 / 1024

        # ONNX 변환
        fp32_path = output_dir / f'rf_pruned_n{n_est}_d{depth}_fp32.onnx'
        fp16_path = output_dir / f'rf_pruned_n{n_est}_d{depth}_fp16.onnx'
        fp32_size, fp16_size = to_onnx_fp16(model, n_features, fp32_path, fp16_path)

        # ONNX 평가
        fp16_metrics, _ = evaluate_onnx(fp16_path, X_test, y_test)

        # 레이턴시
        lat = benchmark_latency(fp16_path, X_test, n_iter=500)

        mape_diff = fp16_metrics['MAPE'] - original_metrics['MAPE']
        size_reduction = (1 - fp16_size / original_size) * 100

        results.append({
            'label': label,
            'n_estimators': n_est,
            'max_depth': depth,
            'pkl_size_mb': pkl_size,
            'fp32_size_mb': fp32_size,
            'fp16_size_mb': fp16_size,
            'mape': fp16_metrics['MAPE'],
            'mae': fp16_metrics['MAE'],
            'rmse': fp16_metrics['RMSE'],
            'mape_diff': mape_diff,
            'size_reduction': size_reduction,
            'train_time_s': train_time,
            'latency_mean_ms': lat['mean_ms'],
            'latency_p99_ms': lat['p99_ms'],
            'fp16_path': str(fp16_path),
        })

        print(f"    pkl: {pkl_size:.2f}MB → ONNX fp16: {fp16_size:.2f}MB ({size_reduction:.1f}% 감소)")
        print(f"    MAPE: {fp16_metrics['MAPE']:.4f}% (원본 대비 {mape_diff:+.4f}%)")
        print(f"    Latency: {lat['mean_ms']:.4f}ms (P99: {lat['p99_ms']:.4f}ms)")

    # 결과 비교
    print("\n" + "=" * 90)
    print("전체 비교 결과")
    print("=" * 90)

    header = f"{'Model':<22} {'pkl(MB)':<10} {'ONNX16(MB)':<12} {'감소율':<10} {'MAPE(%)':<10} {'Δ MAPE':<10} {'Lat(ms)':<10}"
    print(header)
    print("-" * 90)

    # 원본
    print(f"{'Original(n=100,d=∞)':<22} {original_size:<10.2f} {'137.63':<12} {'-':<10} {original_metrics['MAPE']:<10.4f} {'-':<10} {'23.95':<10}")

    for r in results:
        print(f"{r['label']:<22} {r['pkl_size_mb']:<10.2f} {r['fp16_size_mb']:<12.2f} "
              f"{r['size_reduction']:<10.1f}% {r['mape']:<10.4f} {r['mape_diff']:<+10.4f} "
              f"{r['latency_mean_ms']:<10.4f}")

    # 최적 모델 선정
    print("\n" + "=" * 90)
    print("최적 모델 선정")
    print("=" * 90)

    # MAPE 2% 미만이면서 크기가 가장 작은 모델
    viable = [r for r in results if r['mape'] < 2.0]
    if viable:
        best = min(viable, key=lambda x: x['fp16_size_mb'])
        print(f"\n  최적 모델: {best['label']}")
        print(f"  ONNX fp16 크기: {best['fp16_size_mb']:.2f} MB (원본 대비 {best['size_reduction']:.1f}% 감소)")
        print(f"  MAPE: {best['mape']:.4f}% (원본 대비 {best['mape_diff']:+.4f}%)")
        print(f"  추론 레이턴시: {best['latency_mean_ms']:.4f}ms")
        print(f"  파일: {best['fp16_path']}")
    else:
        print("  MAPE < 2% 조건을 만족하는 모델이 없습니다.")

    # 리포트 저장
    report_path = output_dir / 'rf_pruning_report.txt'
    with open(report_path, 'w') as f:
        f.write("Random Forest Tree Pruning + ONNX Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original: {original_size:.2f}MB, MAPE: {original_metrics['MAPE']:.4f}%\n\n")
        for r in results:
            f.write(f"{r['label']}:\n")
            f.write(f"  pkl:       {r['pkl_size_mb']:.2f} MB\n")
            f.write(f"  ONNX fp32: {r['fp32_size_mb']:.2f} MB\n")
            f.write(f"  ONNX fp16: {r['fp16_size_mb']:.2f} MB ({r['size_reduction']:.1f}% reduction)\n")
            f.write(f"  MAPE:      {r['mape']:.4f}% ({r['mape_diff']:+.4f}%)\n")
            f.write(f"  MAE:       {r['mae']:.4f}\n")
            f.write(f"  RMSE:      {r['rmse']:.4f}\n")
            f.write(f"  Latency:   {r['latency_mean_ms']:.4f}ms (P99: {r['latency_p99_ms']:.4f}ms)\n")
            f.write(f"  Train:     {r['train_time_s']:.2f}s\n\n")

    print(f"\n  리포트 저장: {report_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
