#!/usr/bin/env python3
"""
XGBoost ONNX 변환 및 float16 양자화

기존 XGBoost 모델(439KB)을 ONNX로 변환하고 float16 양자화를 적용합니다.

사용법:
    python scripts/optimize_xgb_onnx.py
"""

import time
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import onnx
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxconverter_common import float16
import onnxruntime as ort

warnings.filterwarnings('ignore')


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


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
    parser = argparse.ArgumentParser(description="XGBoost ONNX 변환 및 float16 양자화")
    parser.add_argument('--input', type=str,
                        default='results/xgboost/xgboost_model.pkl',
                        help='XGBoost 모델 경로')
    parser.add_argument('--test-data', type=str,
                        default='data/processed/test_processed.csv')
    parser.add_argument('--output-dir', type=str,
                        default='results/optimized')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("XGBoost ONNX 변환 및 float16 양자화")
    print("=" * 90)

    # 1. 데이터 로드
    print("\n[Step 1] 테스트 데이터 로드")
    test_df = pd.read_csv(args.test_data)
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    n_features = X_test.shape[1]
    print(f"  샘플 수: {len(y_test):,}, Features: {n_features}")

    # 2. XGBoost 모델 로드
    print("\n[Step 2] XGBoost 모델 로드")
    model = joblib.load(args.input)
    pkl_size = Path(args.input).stat().st_size / 1024 / 1024
    print(f"  크기: {pkl_size:.4f} MB ({pkl_size * 1024:.1f} KB)")

    # sklearn 평가
    y_pred_sklearn = model.predict(X_test)
    sklearn_metrics = calculate_metrics(y_test, y_pred_sklearn)
    print(f"  MAPE: {sklearn_metrics['MAPE']:.4f}%")

    # 3. ONNX 변환 (float32)
    print("\n[Step 3] ONNX (float32) 변환")
    fp32_path = output_dir / 'xgb_model_fp32.onnx'

    # XGBoost ONNX 변환은 feature 이름이 'f%d' 형식이어야 함
    booster = model.get_booster()
    booster.feature_names = [f'f{i}' for i in range(n_features)]

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=15)
    onnx.save(onnx_model, str(fp32_path))
    fp32_size = fp32_path.stat().st_size / 1024 / 1024
    print(f"  크기: {fp32_size:.4f} MB ({fp32_size * 1024:.1f} KB)")

    # float32 평가
    sess = ort.InferenceSession(str(fp32_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    y_pred_fp32 = sess.run(None, {input_name: X_test.astype(np.float32)})[0].flatten()
    fp32_metrics = calculate_metrics(y_test, y_pred_fp32)
    print(f"  MAPE: {fp32_metrics['MAPE']:.4f}%")

    # 4. float16 양자화
    print("\n[Step 4] float16 양자화")
    fp16_path = output_dir / 'xgb_model_fp16.onnx'
    model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))
    fp16_size = fp16_path.stat().st_size / 1024 / 1024
    print(f"  크기: {fp16_size:.4f} MB ({fp16_size * 1024:.1f} KB)")

    # float16 평가
    sess16 = ort.InferenceSession(str(fp16_path), providers=['CPUExecutionProvider'])
    y_pred_fp16 = sess16.run(None, {sess16.get_inputs()[0].name: X_test.astype(np.float32)})[0].flatten()
    fp16_metrics = calculate_metrics(y_test, y_pred_fp16)
    print(f"  MAPE: {fp16_metrics['MAPE']:.4f}%")

    # 5. 레이턴시 벤치마크
    print("\n[Step 5] 레이턴시 벤치마크 (1000회)")

    # sklearn
    sample = X_test[:1]
    lat_sklearn = []
    for _ in range(1000):
        start = time.perf_counter()
        model.predict(sample)
        lat_sklearn.append((time.perf_counter() - start) * 1000)

    lat_fp32 = benchmark_latency(fp32_path, X_test)
    lat_fp16 = benchmark_latency(fp16_path, X_test)

    # 6. 결과 비교
    print("\n" + "=" * 90)
    print("XGBoost 모델 비교 결과")
    print("=" * 90)

    header = f"{'Model':<22} {'Size(KB)':<12} {'MAPE(%)':<12} {'MAE':<12} {'RMSE':<12} {'Lat Mean(ms)':<14} {'Lat P99(ms)':<12}"
    print(header)
    print("-" * 90)

    print(f"{'Original (pkl)':<22} {pkl_size*1024:<12.1f} {sklearn_metrics['MAPE']:<12.4f} "
          f"{sklearn_metrics['MAE']:<12.4f} {sklearn_metrics['RMSE']:<12.4f} "
          f"{np.mean(lat_sklearn):<14.4f} {np.percentile(lat_sklearn, 99):<12.4f}")

    print(f"{'ONNX (float32)':<22} {fp32_size*1024:<12.1f} {fp32_metrics['MAPE']:<12.4f} "
          f"{fp32_metrics['MAE']:<12.4f} {fp32_metrics['RMSE']:<12.4f} "
          f"{lat_fp32['mean_ms']:<14.4f} {lat_fp32['p99_ms']:<12.4f}")

    print(f"{'ONNX (float16)':<22} {fp16_size*1024:<12.1f} {fp16_metrics['MAPE']:<12.4f} "
          f"{fp16_metrics['MAE']:<12.4f} {fp16_metrics['RMSE']:<12.4f} "
          f"{lat_fp16['mean_ms']:<14.4f} {lat_fp16['p99_ms']:<12.4f}")

    # 크기 변화
    print(f"\n크기 변화:")
    print(f"  pkl → ONNX fp32: {pkl_size*1024:.1f}KB → {fp32_size*1024:.1f}KB")
    print(f"  pkl → ONNX fp16: {pkl_size*1024:.1f}KB → {fp16_size*1024:.1f}KB")

    # 정확도 변화
    print(f"\n정확도 변화 (MAPE):")
    print(f"  ONNX fp32: {fp32_metrics['MAPE'] - sklearn_metrics['MAPE']:+.4f}%")
    print(f"  ONNX fp16: {fp16_metrics['MAPE'] - sklearn_metrics['MAPE']:+.4f}%")

    # RF 원본과 비교
    rf_pkl_size = Path('results/random_forest/random_forest_model.pkl').stat().st_size / 1024 / 1024
    print(f"\nRandom Forest 원본 대비:")
    print(f"  RF Original: {rf_pkl_size:.2f}MB → XGBoost ONNX fp16: {fp16_size*1024:.1f}KB")
    print(f"  크기 감소: {(1 - fp16_size / rf_pkl_size) * 100:.1f}%")

    # 리포트 저장
    report_path = output_dir / 'xgb_onnx_report.txt'
    with open(report_path, 'w') as f:
        f.write("XGBoost ONNX Optimization Report\n")
        f.write("=" * 50 + "\n\n")
        for label, size_kb, metrics, lat_mean, lat_p99 in [
            ('Original (pkl)', pkl_size*1024, sklearn_metrics, np.mean(lat_sklearn), np.percentile(lat_sklearn, 99)),
            ('ONNX (float32)', fp32_size*1024, fp32_metrics, lat_fp32['mean_ms'], lat_fp32['p99_ms']),
            ('ONNX (float16)', fp16_size*1024, fp16_metrics, lat_fp16['mean_ms'], lat_fp16['p99_ms']),
        ]:
            f.write(f"{label}:\n")
            f.write(f"  Size:    {size_kb:.1f} KB\n")
            f.write(f"  MAPE:    {metrics['MAPE']:.4f}%\n")
            f.write(f"  MAE:     {metrics['MAE']:.4f}\n")
            f.write(f"  RMSE:    {metrics['RMSE']:.4f}\n")
            f.write(f"  Latency: {lat_mean:.4f}ms (P99: {lat_p99:.4f}ms)\n\n")

    print(f"\n리포트 저장: {report_path}")

    print("\n" + "=" * 90)
    print("최적화 완료!")
    print("=" * 90)
    print(f"\n저장된 파일:")
    print(f"  - {fp32_path}")
    print(f"  - {fp16_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
