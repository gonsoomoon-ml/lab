"""
03_baseline_benchmark.py - Baseline benchmark (CUDA + FP32)

Measures baseline performance metrics:
- Latency (ms)
- Throughput (RPS)
- Accuracy (mAP@50, mAP@50:95)
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    # Configure baseline benchmark
    config = BenchmarkConfig(
        experiment_name="03_baseline_benchmark",
        device="cuda",
        dtype="float32",
    )

    # Run benchmark
    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
