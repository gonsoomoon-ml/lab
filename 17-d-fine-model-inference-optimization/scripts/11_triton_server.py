"""
11_triton_server.py - Triton Inference Server benchmark

Sets up model repository and benchmarks Triton Inference Server.
Supports both ONNX Runtime and TensorRT backends.

For production, TensorRT backend is recommended for best performance.

Requirements:
    - Docker with nvidia-docker runtime
    - tritonclient[all] (installed via pyproject.toml)

Usage:
    1. Run this script to setup model repository
    2. Start Triton Server with the provided Docker command
    3. Run this script again to benchmark

Backends:
    - ONNX Runtime: Uses ONNX model directly (simpler, slower)
    - TensorRT: Converts ONNX to TensorRT engine (recommended for production)
"""

import sys
sys.path.append("script")

import os
import shutil
import time
import json
import numpy as np
from utils.benchmark import (
    BenchmarkConfig,
    print_config,
    prepare_inputs,
    export_to_onnx,
    load_model,
    PROJECT_ROOT,
)


def setup_triton_onnx_repository(config, batch_size=4):
    """Create Triton model repository with ONNX Runtime backend."""
    onnx_model_path = os.path.join(PROJECT_ROOT, "output", "onnx", f"dfine_x_coco_batch{batch_size}.onnx")
    triton_repo = os.path.join(PROJECT_ROOT, "output", "triton_model_repository")
    model_dir = os.path.join(triton_repo, "dfine_onnx", "1")

    # Export ONNX model if not exists
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model (batch={batch_size}) not found. Exporting...")
        model, image_processor, device = load_model(config)
        inputs, _ = prepare_inputs(
            config.test_image_path,
            image_processor,
            device,
            "float32",
        )
        # Repeat to create batch
        inputs = {k: v.repeat(batch_size, 1, 1, 1) if v.dim() == 4 else v for k, v in inputs.items()}
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        export_to_onnx(model, inputs, onnx_model_path)

    # Create directory structure
    os.makedirs(model_dir, exist_ok=True)

    # Copy ONNX model
    model_dst = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_dst):
        print(f"Copying ONNX model to: {model_dst}")
        shutil.copy2(onnx_model_path, model_dst)

    # Create config.pbtxt for ONNX Runtime
    config_content = f"""name: "dfine_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 3, 640, 640 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 300, 80 ]
  }},
  {{
    name: "pred_boxes"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 300, 4 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    config_path = os.path.join(triton_repo, "dfine_onnx", "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created ONNX config: {config_path}")

    return triton_repo


def setup_triton_tensorrt_repository(config, batch_size=4):
    """Create Triton model repository with TensorRT backend.

    TensorRT will compile the ONNX model to optimized TensorRT engine on first load.
    This is the recommended backend for production deployments.
    """
    onnx_model_path = os.path.join(PROJECT_ROOT, "output", "onnx", f"dfine_x_coco_batch{batch_size}.onnx")
    triton_repo = os.path.join(PROJECT_ROOT, "output", "triton_model_repository")
    model_dir = os.path.join(triton_repo, "dfine_trt", "1")

    # Export ONNX model if not exists
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model (batch={batch_size}) not found. Exporting...")
        model, image_processor, device = load_model(config)
        inputs, _ = prepare_inputs(
            config.test_image_path,
            image_processor,
            device,
            "float32",
        )
        # Repeat to create batch
        inputs = {k: v.repeat(batch_size, 1, 1, 1) if v.dim() == 4 else v for k, v in inputs.items()}
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        export_to_onnx(model, inputs, onnx_model_path)

    # Create directory structure
    os.makedirs(model_dir, exist_ok=True)

    # Copy ONNX model (Triton will convert to TensorRT on first load)
    model_dst = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(model_dst):
        print(f"Copying ONNX model to: {model_dst}")
        shutil.copy2(onnx_model_path, model_dst)

    # Create config.pbtxt for TensorRT backend
    # Triton will automatically convert ONNX to TensorRT engine
    config_content = f"""name: "dfine_trt"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP16
    dims: [ {batch_size}, 3, 640, 640 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP16
    dims: [ {batch_size}, 300, 80 ]
  }},
  {{
    name: "pred_boxes"
    data_type: TYPE_FP16
    dims: [ {batch_size}, 300, 4 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

# TensorRT optimization settings
optimization {{
  cuda {{
    graphs: true
  }}
}}
"""

    config_path = os.path.join(triton_repo, "dfine_trt", "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created TensorRT config: {config_path}")

    return triton_repo


def convert_onnx_to_tensorrt(onnx_path, trt_path, batch_size=4, fp16=True):
    """Convert ONNX model to TensorRT engine using trtexec."""
    import subprocess

    print(f"Converting ONNX to TensorRT engine...")
    print(f"  Input: {onnx_path}")
    print(f"  Output: {trt_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  FP16: {fp16}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        f"--shapes=pixel_values:{batch_size}x3x640x640",
    ]

    if fp16:
        cmd.append("--fp16")

    # Add workspace size (4GB)
    cmd.append("--workspace=4096")

    print(f"\nRunning: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"trtexec failed:\n{result.stderr}")
            return False
        print("TensorRT conversion successful!")
        return True
    except FileNotFoundError:
        print("trtexec not found. Install TensorRT or use Triton's auto-conversion.")
        return False
    except subprocess.TimeoutExpired:
        print("TensorRT conversion timed out (>10 minutes)")
        return False


def setup_triton_trt_engine_repository(config, batch_size=4):
    """Create Triton model repository with TensorRT engine.

    Uses ONNX Runtime with TensorRT execution provider for automatic optimization.
    This is simpler than building .plan files and gives good performance.
    """
    onnx_model_path = os.path.join(PROJECT_ROOT, "output", "onnx", f"dfine_x_coco_batch{batch_size}.onnx")
    triton_repo = os.path.join(PROJECT_ROOT, "output", "triton_model_repository")
    model_dir = os.path.join(triton_repo, "dfine_trt", "1")

    # Export ONNX model if not exists
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model (batch={batch_size}) not found. Exporting...")
        model, image_processor, device = load_model(config)
        inputs, _ = prepare_inputs(
            config.test_image_path,
            image_processor,
            device,
            "float32",
        )
        # Repeat to create batch
        inputs = {k: v.repeat(batch_size, 1, 1, 1) if v.dim() == 4 else v for k, v in inputs.items()}
        os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
        export_to_onnx(model, inputs, onnx_model_path)

    # Create directory structure
    os.makedirs(model_dir, exist_ok=True)

    # Copy ONNX model
    onnx_dst = os.path.join(model_dir, "model.onnx")
    if not os.path.exists(onnx_dst):
        print(f"Copying ONNX model to: {onnx_dst}")
        shutil.copy2(onnx_model_path, onnx_dst)

    # Create config.pbtxt for ONNX Runtime with TensorRT execution provider
    # This uses ONNX Runtime but accelerates with TensorRT under the hood
    config_content = f"""name: "dfine_trt"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 3, 640, 640 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 300, 80 ]
  }},
  {{
    name: "pred_boxes"
    data_type: TYPE_FP32
    dims: [ {batch_size}, 300, 4 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]

# Use TensorRT execution provider for optimization
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [
      {{
        name : "tensorrt"
        parameters {{ key: "precision_mode" value: "FP16" }}
        parameters {{ key: "max_workspace_size_bytes" value: "4294967296" }}
      }}
    ]
  }}
}}
"""

    config_path = os.path.join(triton_repo, "dfine_trt", "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created TensorRT-optimized ONNX config: {config_path}")

    return triton_repo


def setup_triton_plan_repository(plan_path, batch_size=1, precision="int8"):
    """Create Triton model repository with pre-built TensorRT .plan file.

    Args:
        plan_path: Path to existing TensorRT .plan file
        batch_size: Batch size the plan was built with
        precision: Precision mode (fp16, int8)
    """
    triton_repo = os.path.join(PROJECT_ROOT, "output", "triton_model_repository")
    model_name = f"dfine_{precision}"
    model_dir = os.path.join(triton_repo, model_name, "1")

    if not os.path.exists(plan_path):
        print(f"ERROR: TensorRT plan not found: {plan_path}")
        return None, None

    # Create directory structure
    os.makedirs(model_dir, exist_ok=True)

    # Copy .plan file
    plan_dst = os.path.join(model_dir, "model.plan")
    if not os.path.exists(plan_dst) or os.path.getmtime(plan_path) > os.path.getmtime(plan_dst):
        print(f"Copying TensorRT plan to: {plan_dst}")
        shutil.copy2(plan_path, plan_dst)
    else:
        print(f"Using existing plan: {plan_dst}")

    # INT8 engines built by trtexec typically use FP32 for I/O
    # The INT8 quantization only affects internal computations
    input_dtype = "TYPE_FP32"
    output_dtype = "TYPE_FP32"

    # Create config.pbtxt for TensorRT plan
    config_content = f"""name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: 0

input [
  {{
    name: "pixel_values"
    data_type: {input_dtype}
    dims: [ {batch_size}, 3, 640, 640 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: {output_dtype}
    dims: [ {batch_size}, 300, 80 ]
  }},
  {{
    name: "pred_boxes"
    data_type: {output_dtype}
    dims: [ {batch_size}, 300, 4 ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    config_path = os.path.join(triton_repo, model_name, "config.pbtxt")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Created config: {config_path}")
    print(f"Model name: {model_name}")
    print(f"Precision: {precision.upper()}")

    return triton_repo, model_name


def print_triton_instructions(triton_repo, model_name="dfine_onnx", use_trt10=False):
    """Print instructions for running Triton Server."""
    print("\n" + "=" * 60)
    print("TRITON INFERENCE SERVER SETUP")
    print("=" * 60)

    # Use Triton 25.10 for TensorRT 10.x plans (built locally)
    if use_trt10:
        triton_image = "nvcr.io/nvidia/tritonserver:25.10-py3"
        trt_version = "10.13.3.9"
    else:
        triton_image = "nvcr.io/nvidia/tritonserver:24.01-py3"
        trt_version = "8.6.1"

    print(f"""
Model repository created at:
  {triton_repo}

Triton Image: {triton_image}
TensorRT Version: {trt_version}

To start Triton Server with Docker:

  docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
    -v {triton_repo}:/models \\
    {triton_image} \\
    tritonserver --model-repository=/models

Once Triton is running, run this script again to benchmark.

Endpoints:
  - HTTP: http://localhost:8000/v2/models/{model_name}/infer
  - gRPC: localhost:8001
  - Metrics: http://localhost:8002/metrics
""")


def run_triton_benchmark(config, model_name="dfine_onnx", batch_size=4):
    """Benchmark Triton Inference Server using HTTP client."""
    try:
        import tritonclient.http as httpclient
    except ImportError:
        print("ERROR: tritonclient not installed.")
        print("Install with: pip install tritonclient[all]")
        return None

    # Check if Triton is running
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        if not client.is_server_live():
            print("Triton Server is not running.")
            return None
        print("Connected to Triton Server")
    except Exception as e:
        print(f"Cannot connect to Triton Server: {e}")
        print("Please start Triton Server first (see instructions above)")
        return None

    # Check if model is loaded
    try:
        if not client.is_model_ready(model_name):
            print(f"Model '{model_name}' is not ready. Check Triton logs.")
            # List available models
            models = client.get_model_repository_index()
            print(f"Available models: {[m['name'] for m in models]}")
            return None
    except Exception as e:
        print(f"Cannot check model status: {e}")
        return None

    # Prepare input
    from transformers import AutoImageProcessor
    from transformers.image_utils import load_image

    image_processor = AutoImageProcessor.from_pretrained(config.model_name, use_fast=True)
    image = load_image(config.test_image_path)
    inputs = image_processor(images=image, return_tensors="pt")
    # Repeat to batch size
    pixel_values = inputs["pixel_values"].repeat(batch_size, 1, 1, 1).numpy()

    # Both models use FP32 input (TRT EP handles FP16 conversion internally)
    pixel_values = pixel_values.astype(np.float32)
    dtype_str = "FP32"

    # Create Triton input
    triton_input = httpclient.InferInput("pixel_values", pixel_values.shape, dtype_str)
    triton_input.set_data_from_numpy(pixel_values)

    # Warmup
    print(f"\nModel: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Input dtype: {dtype_str}")
    print(f"\nWarmup ({config.num_warmup} iterations)...")
    for _ in range(config.num_warmup):
        _ = client.infer(model_name, inputs=[triton_input])

    # Benchmark
    print(f"Benchmarking latency ({config.num_iterations} iterations)...")
    latencies = []
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        _ = client.infer(model_name, inputs=[triton_input])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latency_results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }

    throughput_ips = batch_size * 1000 / latency_results["mean_ms"]

    # Print results
    print(f"\n{'='*50}")
    print(f"TRITON SERVER RESULTS ({model_name}, batch={batch_size})")
    print(f"{'='*50}")
    print(f"Mean latency: {latency_results['mean_ms']:.2f} ms")
    print(f"Std latency: {latency_results['std_ms']:.2f} ms")
    print(f"P50 latency: {latency_results['p50_ms']:.2f} ms")
    print(f"P95 latency: {latency_results['p95_ms']:.2f} ms")
    print(f"P99 latency: {latency_results['p99_ms']:.2f} ms")
    print(f"Throughput: {throughput_ips:.2f} images/sec")

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, "output", f"{config.experiment_name}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    results_data = {
        "config": {
            "experiment_name": config.experiment_name,
            "server": "triton",
            "backend": model_name,
            "batch_size": batch_size,
        },
        "latency": latency_results,
        "throughput": {"images_per_sec": throughput_ips},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results_data


def main(trt_file=None, precision="fp16", batch_size=4):
    config = BenchmarkConfig(
        experiment_name="11_triton_server",
        device="cuda",
        dtype="float32",
    )

    print_config(config)

    # If a pre-built TensorRT plan file is provided, use it
    if trt_file:
        print("\n" + "=" * 60)
        print(f"Setting up Triton with pre-built TensorRT plan ({precision.upper()})...")
        print("=" * 60)

        triton_repo, model_name = setup_triton_plan_repository(trt_file, batch_size, precision)
        if triton_repo is None:
            return None

        # Try to benchmark if Triton is running
        print("\n" + "=" * 60)
        print("Benchmarking...")
        print("=" * 60)

        results = None
        try:
            results = run_triton_benchmark(config, model_name, batch_size)
        except Exception as e:
            print(f"Benchmark failed for {model_name}: {e}")

        if results is None:
            # Triton not running, print setup instructions
            # Use TRT10 flag since local plans are built with TensorRT 10.x
            print_triton_instructions(triton_repo, model_name, use_trt10=True)

        return results

    # Default: Setup both ONNX and TensorRT model repositories
    print("\n" + "=" * 60)
    print("Setting up Triton model repositories...")
    print("=" * 60)

    print("\n1. Setting up ONNX Runtime backend...")
    triton_repo = setup_triton_onnx_repository(config, batch_size)

    print("\n2. Setting up TensorRT backend...")
    setup_triton_trt_engine_repository(config, batch_size)

    # Try to benchmark if Triton is running
    print("\n" + "=" * 60)
    print("Benchmarking...")
    print("=" * 60)

    # Try TensorRT first (recommended), then fall back to ONNX
    results = None
    for model_name in ["dfine_trt", "dfine_onnx"]:
        try:
            results = run_triton_benchmark(config, model_name, batch_size)
            if results:
                break
        except Exception as e:
            print(f"Benchmark failed for {model_name}: {e}")
            continue

    if results is None:
        # Triton not running, print setup instructions
        print_triton_instructions(triton_repo)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triton Inference Server benchmark")
    parser.add_argument("--trt-file", type=str, default=None,
                        help="Path to pre-built TensorRT .plan file")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "int8"],
                        help="Precision mode of the TensorRT plan")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size the plan was built with")
    args = parser.parse_args()

    main(trt_file=args.trt_file, precision=args.precision, batch_size=args.batch_size)
