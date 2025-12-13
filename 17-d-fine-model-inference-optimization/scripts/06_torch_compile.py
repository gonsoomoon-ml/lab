"""
06_torch_compile.py - PyTorch 2.0+ torch.compile() optimization

Uses default compile mode for balanced optimization.
Note: First run includes compilation time. Warmup handles this.
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    config = BenchmarkConfig(
        experiment_name="06_torch_compile",
        device="cuda",
        dtype="float32",
        use_compile=True,
        compile_mode="default",
    )

    results = run_benchmark(config)

    return results


if __name__ == "__main__":
    main()
