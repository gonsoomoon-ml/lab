"""
04_num_queries.py - Test num_queries optimization (300, 150, 100)

Fewer object query candidates = faster decoder processing.
Tests the trade-off between speed and detection accuracy.
"""

import sys
sys.path.append("script")

from utils.benchmark import BenchmarkConfig, run_benchmark


def main():
    # Test different num_queries values
    num_queries_values = [300, 150, 100]

    all_results = {}

    for num_queries in num_queries_values:
        print(f"\n{'#'*60}")
        print(f"# Testing num_queries = {num_queries}")
        print(f"{'#'*60}")

        config = BenchmarkConfig(
            experiment_name=f"04_num_queries_{num_queries}",
            device="cuda",
            dtype="float32",
            num_queries=num_queries,
        )

        results = run_benchmark(config)
        all_results[num_queries] = results

    # Print comparison summary
    print(f"\n{'='*70}")
    print("NUM_QUERIES COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'num_queries':<15} {'Latency (ms)':<15} {'RPS':<10} {'mAP@50':<10} {'mAP@50:95':<10}")
    print("-" * 70)

    for num_queries, results in all_results.items():
        latency = results["latency"]["mean_ms"]
        rps = results["rps"]
        map_50 = results["accuracy"].get("mAP_50")
        map_50_95 = results["accuracy"].get("mAP_50_95")

        if map_50 is not None:
            print(f"{num_queries:<15} {latency:<15.2f} {rps:<10.2f} {map_50:<10.4f} {map_50_95:<10.4f}")
        else:
            print(f"{num_queries:<15} {latency:<15.2f} {rps:<10.2f} {'N/A':<10} {'N/A':<10}")

    return all_results


if __name__ == "__main__":
    main()
