"""
05_gpu_fp16.py - FP16 (Half Precision) optimization

Uses torch.float16 for faster computation on Tesla T4.
Expected: 1.5-2x speedup with minimal accuracy loss (<1% mAP).
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    config = BenchmarkConfig(
        experiment_name="05_gpu_fp16",
        device="cuda",
        dtype="float16",
    )

    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
