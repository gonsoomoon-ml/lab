"""
07_tensorrt.py - TensorRT optimization via torch.compile backend

Uses torch.compile with TensorRT backend for GPU optimization.
Note: TensorRT compilation may take a few minutes on first run.

Requirements:
    torch-tensorrt (installed via pyproject.toml)
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    config = BenchmarkConfig(
        experiment_name="07_tensorrt",
        device="cuda",
        dtype="float16",  # TensorRT works best with FP16
        use_compile=True,
        compile_backend="torch_tensorrt",
    )

    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
