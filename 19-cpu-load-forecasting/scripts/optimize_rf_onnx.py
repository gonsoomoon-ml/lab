#!/usr/bin/env python3
"""
Random Forest ONNX 변환 및 float16 양자화

기존 Random Forest 모델을 ONNX로 변환하고 float16 양자화를 적용하여
네트워크 장비 배포용 경량 모델을 생성합니다.

사용법:
    python scripts/optimize_rf_onnx.py
    python scripts/optimize_rf_onnx.py --input results/random_forest/random_forest_model.pkl
"""

import time
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ONNX
import onnx
from onnx import numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


def load_sklearn_model(model_path):
    """sklearn 모델 로드"""
    print(f"  모델 로드: {model_path}")
    model = joblib.load(model_path)
    size_mb = Path(model_path).stat().st_size / 1024 / 1024
    print(f"  크기: {size_mb:.2f} MB")
    return model, size_mb


def convert_to_onnx(model, n_features, output_path):
    """sklearn Random Forest → ONNX (float32) 변환"""
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=15)
    onnx.save(onnx_model, str(output_path))

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  ONNX (float32) 저장: {output_path}")
    print(f"  크기: {size_mb:.2f} MB")
    return onnx_model, size_mb


def convert_to_float16(onnx_model_path, output_path):
    """ONNX float32 → float16 양자화"""
    from onnxconverter_common import float16

    model_fp32 = onnx.load(str(onnx_model_path))
    model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"  ONNX (float16) 저장: {output_path}")
    print(f"  크기: {size_mb:.2f} MB")
    return model_fp16, size_mb


def evaluate_sklearn(model, X_test, y_test):
    """sklearn 모델 평가"""
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = (time.perf_counter() - start) * 1000

    metrics = calculate_metrics(y_test, y_pred)
    metrics['inference_time_ms'] = elapsed
    return metrics, y_pred


def evaluate_onnx(model_path, X_test, y_test):
    """ONNX 모델 평가"""
    sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    X_float32 = X_test.astype(np.float32)

    start = time.perf_counter()
    y_pred = sess.run(None, {input_name: X_float32})[0].flatten()
    elapsed = (time.perf_counter() - start) * 1000

    metrics = calculate_metrics(y_test, y_pred)
    metrics['inference_time_ms'] = elapsed
    return metrics, y_pred


def calculate_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE 계산"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def benchmark_latency(model_path, X_sample, n_iterations=1000, is_sklearn=False, sklearn_model=None):
    """추론 레이턴시 벤치마크 (단일 샘플 기준)"""
    sample = X_sample[:1].astype(np.float32)
    latencies = []

    if is_sklearn:
        for _ in range(n_iterations):
            start = time.perf_counter()
            sklearn_model.predict(sample)
            latencies.append((time.perf_counter() - start) * 1000)
    else:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        for _ in range(n_iterations):
            start = time.perf_counter()
            sess.run(None, {input_name: sample})
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }


def print_comparison(results):
    """비교 결과 출력"""
    print("\n" + "=" * 90)
    print("모델 비교 결과")
    print("=" * 90)

    header = f"{'Model':<25} {'Size(MB)':<12} {'MAPE(%)':<12} {'MAE':<12} {'RMSE':<12} {'Infer(ms)':<12}"
    print(header)
    print("-" * 90)

    for name, r in results.items():
        print(f"{name:<25} {r['size_mb']:<12.2f} {r['metrics']['MAPE']:<12.4f} "
              f"{r['metrics']['MAE']:<12.4f} {r['metrics']['RMSE']:<12.4f} "
              f"{r['metrics']['inference_time_ms']:<12.2f}")

    # 크기 감소율
    original_size = results['Original (pkl)']['size_mb']
    print("\n" + "-" * 90)
    print("크기 감소율 (Original 대비):")
    for name, r in results.items():
        if name == 'Original (pkl)':
            continue
        reduction = (1 - r['size_mb'] / original_size) * 100
        print(f"  {name}: {reduction:.1f}% 감소 ({original_size:.2f}MB → {r['size_mb']:.2f}MB)")

    # 정확도 변화
    original_mape = results['Original (pkl)']['metrics']['MAPE']
    print("\n정확도 변화 (MAPE, Original 대비):")
    for name, r in results.items():
        if name == 'Original (pkl)':
            continue
        diff = r['metrics']['MAPE'] - original_mape
        print(f"  {name}: {diff:+.4f}% ({'악화' if diff > 0 else '개선'})")


def print_latency_comparison(latency_results):
    """레이턴시 벤치마크 결과 출력"""
    print("\n" + "=" * 90)
    print("추론 레이턴시 벤치마크 (단일 샘플, 1000회 반복)")
    print("=" * 90)

    header = f"{'Model':<25} {'Mean(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'P99(ms)':<12}"
    print(header)
    print("-" * 90)

    for name, r in latency_results.items():
        print(f"{name:<25} {r['mean_ms']:<12.4f} {r['p50_ms']:<12.4f} "
              f"{r['p95_ms']:<12.4f} {r['p99_ms']:<12.4f}")


def save_report(results, latency_results, output_path):
    """벤치마크 결과를 텍스트 파일로 저장"""
    with open(output_path, 'w') as f:
        f.write("Random Forest ONNX Optimization Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Model Size & Accuracy]\n")
        for name, r in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Size:  {r['size_mb']:.2f} MB\n")
            f.write(f"  MAPE:  {r['metrics']['MAPE']:.4f}%\n")
            f.write(f"  MAE:   {r['metrics']['MAE']:.4f}\n")
            f.write(f"  RMSE:  {r['metrics']['RMSE']:.4f}\n")

        original_size = results['Original (pkl)']['size_mb']
        f.write(f"\n[Size Reduction]\n")
        for name, r in results.items():
            if name == 'Original (pkl)':
                continue
            reduction = (1 - r['size_mb'] / original_size) * 100
            f.write(f"  {name}: {reduction:.1f}% reduction\n")

        f.write(f"\n[Latency Benchmark (single sample, 1000 iterations)]\n")
        for name, r in latency_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Mean: {r['mean_ms']:.4f} ms\n")
            f.write(f"  P50:  {r['p50_ms']:.4f} ms\n")
            f.write(f"  P95:  {r['p95_ms']:.4f} ms\n")
            f.write(f"  P99:  {r['p99_ms']:.4f} ms\n")

    print(f"\n리포트 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Random Forest ONNX 변환 및 float16 양자화")
    parser.add_argument('--input', type=str,
                        default='results/random_forest/random_forest_model.pkl',
                        help='원본 Random Forest 모델 경로')
    parser.add_argument('--test-data', type=str,
                        default='data/processed/test_processed.csv',
                        help='테스트 데이터 경로')
    parser.add_argument('--output-dir', type=str,
                        default='results/optimized',
                        help='최적화 모델 저장 디렉토리')
    parser.add_argument('--benchmark-iterations', type=int, default=1000,
                        help='레이턴시 벤치마크 반복 횟수')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("Random Forest ONNX 변환 및 float16 양자화")
    print("=" * 90)

    # 1. 테스트 데이터 로드
    print("\n[Step 1] 테스트 데이터 로드")
    test_df = pd.read_csv(args.test_data)
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    n_features = X_test.shape[1]
    print(f"  샘플 수: {len(y_test):,}, Features: {n_features}")

    # 2. 원본 sklearn 모델 로드 및 평가
    print("\n[Step 2] 원본 sklearn 모델 로드")
    sklearn_model, pkl_size = load_sklearn_model(args.input)
    print("  평가 중...")
    sklearn_metrics, _ = evaluate_sklearn(sklearn_model, X_test, y_test)
    print(f"  MAPE: {sklearn_metrics['MAPE']:.4f}%")

    # 3. ONNX 변환 (float32)
    print("\n[Step 3] ONNX (float32) 변환")
    onnx_fp32_path = output_dir / 'rf_model_fp32.onnx'
    _, fp32_size = convert_to_onnx(sklearn_model, n_features, onnx_fp32_path)
    print("  평가 중...")
    fp32_metrics, _ = evaluate_onnx(onnx_fp32_path, X_test, y_test)
    print(f"  MAPE: {fp32_metrics['MAPE']:.4f}%")

    # 4. float16 양자화
    print("\n[Step 4] float16 양자화")
    onnx_fp16_path = output_dir / 'rf_model_fp16.onnx'
    _, fp16_size = convert_to_float16(onnx_fp32_path, onnx_fp16_path)
    print("  평가 중...")
    fp16_metrics, _ = evaluate_onnx(onnx_fp16_path, X_test, y_test)
    print(f"  MAPE: {fp16_metrics['MAPE']:.4f}%")

    # 5. 레이턴시 벤치마크
    print(f"\n[Step 5] 레이턴시 벤치마크 ({args.benchmark_iterations}회 반복)")
    X_sample = X_test.astype(np.float32)

    print("  Original (sklearn)...")
    lat_sklearn = benchmark_latency(None, X_sample, args.benchmark_iterations,
                                     is_sklearn=True, sklearn_model=sklearn_model)
    print("  ONNX (float32)...")
    lat_fp32 = benchmark_latency(onnx_fp32_path, X_sample, args.benchmark_iterations)
    print("  ONNX (float16)...")
    lat_fp16 = benchmark_latency(onnx_fp16_path, X_sample, args.benchmark_iterations)

    # 6. 결과 비교
    results = {
        'Original (pkl)': {'size_mb': pkl_size, 'metrics': sklearn_metrics},
        'ONNX (float32)': {'size_mb': fp32_size, 'metrics': fp32_metrics},
        'ONNX (float16)': {'size_mb': fp16_size, 'metrics': fp16_metrics},
    }

    latency_results = {
        'Original (pkl)': lat_sklearn,
        'ONNX (float32)': lat_fp32,
        'ONNX (float16)': lat_fp16,
    }

    print_comparison(results)
    print_latency_comparison(latency_results)

    # 7. 리포트 저장
    report_path = output_dir / 'optimization_report.txt'
    save_report(results, latency_results, report_path)

    print("\n" + "=" * 90)
    print("최적화 완료!")
    print("=" * 90)
    print(f"\n저장된 파일:")
    print(f"  - {onnx_fp32_path}")
    print(f"  - {onnx_fp16_path}")
    print(f"  - {report_path}")


if __name__ == "__main__":
    main()
