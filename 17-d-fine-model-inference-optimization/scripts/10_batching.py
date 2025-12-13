"""
10_batching.py - Batch inference optimization

Tests different batch sizes to find optimal throughput.
Larger batches improve GPU utilization but increase memory usage.
Expected: Higher throughput (images/sec) with larger batches.
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    config = BenchmarkConfig(
        experiment_name="10_batching",
        device="cuda",
        dtype="float16",  # Use FP16 for better memory efficiency
        batch_sizes=[1, 2, 4, 8, 16],
    )

    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
