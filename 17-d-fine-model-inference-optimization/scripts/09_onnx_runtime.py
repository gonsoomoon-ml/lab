"""
08_onnx_runtime.py - ONNX Runtime optimization

Exports model to ONNX format and runs inference with ONNX Runtime GPU provider.
Expected: 1.5-2x speedup over baseline.

Requirements:
    onnx, onnxruntime-gpu (installed via pyproject.toml)
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    config = BenchmarkConfig(
        experiment_name="08_onnx_runtime",
        device="cuda",
        dtype="float32",
        use_onnx=True,
    )

    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
