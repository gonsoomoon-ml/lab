"""
07b_build_tensorrt_engine.py - Build TensorRT engine from ONNX model

This script:
1. Exports D-Fine model to ONNX with only final outputs (logits, pred_boxes)
2. Converts ONNX to TensorRT engine using trtexec in Docker
3. Benchmarks the TensorRT engine with latency and mAP evaluation

Requirements:
    - Docker with nvidia-docker runtime
    - TensorRT Docker image: nvcr.io/nvidia/tensorrt:24.01-py3
"""

import sys
sys.path.append("script")

import os
import time
import json
import numpy as np
import torch
import onnx
from utils.benchmark import (
    BenchmarkConfig,
    load_model,
    prepare_inputs,
    PROJECT_ROOT,
)


def export_onnx_final_outputs_only(model, inputs, onnx_path, batch_size=4):
    """Export PyTorch model to ONNX with only final outputs (logits, pred_boxes).

    D-Fine model exports intermediate outputs by default. This function
    exports and then trims the ONNX graph to only keep the final outputs.
    """
    print(f"Exporting model to ONNX (batch={batch_size}): {onnx_path}")

    pixel_values = inputs["pixel_values"]

    # Ensure batch size
    if pixel_values.shape[0] != batch_size:
        pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

    # Export with static batch size for TensorRT compatibility
    temp_path = onnx_path.replace(".onnx", "_temp.onnx")

    torch.onnx.export(
        model,
        (pixel_values,),
        temp_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=18,  # dynamo=True requires opset 18+
        dynamo=True,  # Use new torch.export-based exporter
    )

    # Load and trim to only final outputs
    print("Trimming ONNX graph to only include logits and pred_boxes...")
    onnx_model = onnx.load(temp_path)

    # Keep only first 2 outputs (logits, pred_boxes)
    while len(onnx_model.graph.output) > 2:
        onnx_model.graph.output.remove(onnx_model.graph.output[-1])

    # Update output names to ensure consistency
    onnx_model.graph.output[0].name = "logits"
    onnx_model.graph.output[1].name = "pred_boxes"

    # Run shape inference to fix any shape issues
    from onnx import shape_inference
    onnx_model = shape_inference.infer_shapes(onnx_model)

    # Save trimmed model
    onnx.save(onnx_model, onnx_path)

    # Clean up temp file
    os.remove(temp_path)

    # Verify outputs
    onnx_model = onnx.load(onnx_path)
    print(f"ONNX outputs after trimming:")
    for i, output in enumerate(onnx_model.graph.output):
        dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"  {i}: {output.name} - shape: {dims}")

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


def build_tensorrt_from_onnx(onnx_path, trt_path, batch_size=4, fp16=True):
    """Build TensorRT engine from ONNX using trtexec in Docker."""
    import subprocess

    print(f"\nBuilding TensorRT engine from ONNX...")
    print(f"  ONNX: {onnx_path}")
    print(f"  Output: {trt_path}")
    print(f"  Batch: {batch_size}")
    print(f"  FP16: {fp16}")

    # Use TensorRT container to run trtexec
    onnx_dir = os.path.dirname(onnx_path)
    onnx_filename = os.path.basename(onnx_path)
    trt_dir = os.path.dirname(trt_path)
    trt_filename = os.path.basename(trt_path)

    os.makedirs(trt_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--gpus=1", "--rm",
        "-v", f"{onnx_dir}:/onnx",
        "-v", f"{trt_dir}:/trt",
        "nvcr.io/nvidia/tensorrt:24.01-py3",
        "trtexec",
        f"--onnx=/onnx/{onnx_filename}",
        f"--saveEngine=/trt/{trt_filename}",
        "--memPoolSize=workspace:8192MiB",
    ]

    if fp16:
        cmd.append("--fp16")

    print(f"\nRunning: {' '.join(cmd)}")
    print("\nThis may take 5-10 minutes...\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse trtexec output for performance metrics
    output = result.stdout + result.stderr

    # Extract GPU compute time
    gpu_compute_ms = None
    for line in output.split('\n'):
        if 'GPU Compute Time' in line and 'mean' in line.lower():
            # Parse: GPU Compute Time: min = X ms, max = Y ms, mean = Z ms
            parts = line.split('mean = ')
            if len(parts) > 1:
                gpu_compute_ms = float(parts[1].split(' ms')[0])
                break

    if result.returncode != 0:
        print("TensorRT build failed!")
        print(result.stderr)
        return None, None

    print(f"\nTensorRT engine saved to: {trt_path}")
    if gpu_compute_ms:
        print(f"GPU Compute Time (mean): {gpu_compute_ms:.2f} ms")

    return trt_path, gpu_compute_ms


def run_tensorrt_benchmark_docker(trt_path, num_warmup=10, num_iterations=100):
    """Benchmark TensorRT engine using trtexec in Docker (same TRT version as build)."""
    import subprocess

    print(f"\nBenchmarking TensorRT engine in Docker: {trt_path}")

    trt_dir = os.path.dirname(os.path.abspath(trt_path))
    trt_filename = os.path.basename(trt_path)

    cmd = [
        "docker", "run", "--gpus=1", "--rm",
        "-v", f"{trt_dir}:/trt",
        "nvcr.io/nvidia/tensorrt:24.01-py3",
        "trtexec",
        f"--loadEngine=/trt/{trt_filename}",
        f"--warmUp={num_warmup * 100}",  # warmUp is in ms
        f"--iterations={num_iterations}",
        "--useSpinWait",
    ]

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse trtexec output for performance metrics
    latency_results = {}
    for line in output.split('\n'):
        if 'GPU Compute Time' in line:
            # Parse: GPU Compute Time: min = X ms, max = Y ms, mean = Z ms, median = W ms
            if 'min' in line:
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if 'min' in part:
                        latency_results['min_ms'] = float(part.split('=')[1].replace('ms', '').strip())
                    elif 'max' in part:
                        latency_results['max_ms'] = float(part.split('=')[1].replace('ms', '').strip())
                    elif 'mean' in part:
                        latency_results['mean_ms'] = float(part.split('=')[1].replace('ms', '').strip())
                    elif 'median' in part:
                        latency_results['p50_ms'] = float(part.split('=')[1].replace('ms', '').strip())
        elif 'percentile' in line.lower() and 'latency' in line.lower():
            # Parse percentile lines like: Percentile latency (99%): X ms
            if '99%' in line or '99 %' in line:
                latency_results['p99_ms'] = float(line.split(':')[1].replace('ms', '').strip())
            elif '95%' in line or '95 %' in line:
                latency_results['p95_ms'] = float(line.split(':')[1].replace('ms', '').strip())

    if result.returncode != 0:
        print("Benchmark failed!")
        print(result.stderr)
        return None

    # Fill in missing values
    if 'std_ms' not in latency_results:
        latency_results['std_ms'] = 0.0
    if 'p95_ms' not in latency_results:
        latency_results['p95_ms'] = latency_results.get('p99_ms', latency_results.get('mean_ms', 0))
    if 'p99_ms' not in latency_results:
        latency_results['p99_ms'] = latency_results.get('max_ms', latency_results.get('mean_ms', 0))

    print(f"GPU Compute Time (mean): {latency_results.get('mean_ms', 0):.2f} ms")

    return latency_results


def run_tensorrt_benchmark(trt_path, batch_size=4, num_warmup=10, num_iterations=100):
    """Benchmark TensorRT engine using Python TensorRT runtime."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT Python bindings not available.")
        print("Using trtexec benchmark results instead.")
        return None

    print(f"\nBenchmarking TensorRT engine: {trt_path}")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)

        device_mem = cuda.mem_alloc(size)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({'name': name, 'shape': shape, 'dtype': dtype, 'mem': device_mem})
        else:
            outputs.append({'name': name, 'shape': shape, 'dtype': dtype, 'mem': device_mem})

    # Create input data
    input_shape = inputs[0]['shape']
    input_data = np.random.randn(*input_shape).astype(inputs[0]['dtype'])

    # Warmup
    print(f"Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        cuda.memcpy_htod(inputs[0]['mem'], input_data)
        context.execute_v2(bindings)
        cuda.Context.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    latencies = []
    for _ in range(num_iterations):
        cuda.memcpy_htod(inputs[0]['mem'], input_data)
        cuda.Context.synchronize()

        start = time.perf_counter()
        context.execute_v2(bindings)
        cuda.Context.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def run_tensorrt_accuracy_evaluation(trt_path, config, batch_size=4):
    """Evaluate mAP using TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT Python bindings not available for accuracy evaluation.")
        return {"mAP_50": None, "mAP_50_95": None}

    from transformers import AutoImageProcessor
    from transformers.image_utils import load_image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io

    print(f"\n{'='*50}")
    print(f"ACCURACY EVALUATION ({config.validation_images} images)")
    print(f"{'='*50}")

    # Check if COCO dataset exists
    if not (os.path.exists(config.coco_ann_path) and os.path.exists(config.coco_val_path)):
        print(f"COCO dataset not found at {config.coco_val_path}")
        return {"mAP_50": None, "mAP_50_95": None}

    # Load TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Get binding info
    input_name = engine.get_tensor_name(0)
    input_shape = engine.get_tensor_shape(input_name)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))

    # Allocate device memory
    input_size = int(np.prod(input_shape) * np.dtype(input_dtype).itemsize)
    d_input = cuda.mem_alloc(input_size)

    output_buffers = []
    for i in range(1, engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        d_output = cuda.mem_alloc(size)
        h_output = np.empty(shape, dtype=dtype)
        output_buffers.append({'name': name, 'shape': shape, 'dtype': dtype,
                               'd_mem': d_output, 'h_mem': h_output})

    bindings = [int(d_input)] + [int(buf['d_mem']) for buf in output_buffers]

    # Load image processor and COCO
    image_processor = AutoImageProcessor.from_pretrained(config.model_name, use_fast=False)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_gt = COCO(config.coco_ann_path)
    sys.stdout = old_stdout

    img_ids = coco_gt.getImgIds()[:config.validation_images]
    coco_categories = coco_gt.loadCats(coco_gt.getCatIds())
    coco_cat_ids = [cat['id'] for cat in coco_categories]

    # Load id2label mapping from model config
    from transformers import DFineForObjectDetection
    model_config = DFineForObjectDetection.from_pretrained(config.model_name).config

    results = []

    print(f"Running inference on {len(img_ids)} images...")
    for idx, img_id in enumerate(img_ids):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(img_ids)}")

        # Load image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(config.coco_val_path, img_info['file_name'])
        image = load_image(img_path)

        # Preprocess
        inputs = image_processor(images=image, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(input_dtype)

        # Pad to batch size if needed
        if pixel_values.shape[0] < batch_size:
            pad_shape = list(pixel_values.shape)
            pad_shape[0] = batch_size - pixel_values.shape[0]
            padding = np.zeros(pad_shape, dtype=input_dtype)
            pixel_values = np.concatenate([pixel_values, padding], axis=0)

        # Run inference
        cuda.memcpy_htod(d_input, pixel_values)
        context.execute_v2(bindings)

        # Copy outputs
        for buf in output_buffers:
            cuda.memcpy_dtoh(buf['h_mem'], buf['d_mem'])

        # Get logits and pred_boxes (first image in batch)
        logits = output_buffers[0]['h_mem'][0]  # [300, 80]
        pred_boxes = output_buffers[1]['h_mem'][0]  # [300, 4]

        # Post-process (similar to image_processor.post_process_object_detection)
        scores = 1 / (1 + np.exp(-logits))  # sigmoid
        max_scores = scores.max(axis=1)
        labels = scores.argmax(axis=1)

        # Filter by confidence threshold
        mask = max_scores > config.confidence_threshold
        filtered_scores = max_scores[mask]
        filtered_labels = labels[mask]
        filtered_boxes = pred_boxes[mask]

        # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] absolute
        cx, cy, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
        x1 = (cx - w/2) * image.width
        y1 = (cy - h/2) * image.height
        x2 = (cx + w/2) * image.width
        y2 = (cy + h/2) * image.height

        # Convert to COCO format
        for score, label, x1_, y1_, x2_, y2_ in zip(filtered_scores, filtered_labels, x1, y1, x2, y2):
            coco_cat_id = coco_cat_ids[label]
            results.append({
                "image_id": img_id,
                "category_id": coco_cat_id,
                "bbox": [float(x1_), float(y1_), float(x2_ - x1_), float(y2_ - y1_)],
                "score": float(score)
            })

    # Evaluate
    if results:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        sys.stdout = old_stdout

        map_50_95 = float(coco_eval.stats[0])
        map_50 = float(coco_eval.stats[1])

        print(f"mAP@50:95: {map_50_95:.4f}")
        print(f"mAP@50: {map_50:.4f}")

        return {"mAP_50": map_50, "mAP_50_95": map_50_95}
    else:
        print("No detections found!")
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}


def main(batch_size=4, precision="fp16", run_accuracy=True, benchmark_only=False, trt_file=None, use_docker=False):
    """Main function to build and benchmark TensorRT engine."""

    config = BenchmarkConfig(
        experiment_name="07b_tensorrt_trtexec",
        device="cuda",
        dtype="float32",  # Always export ONNX in FP32, TensorRT handles FP16 conversion
    )

    # Paths
    onnx_dir = os.path.join(PROJECT_ROOT, "output", "onnx_trimmed")
    onnx_path = os.path.join(onnx_dir, f"dfine_x_coco_batch{batch_size}.onnx")
    trt_dir = os.path.join(PROJECT_ROOT, "output", "tensorrt")
    trt_path = os.path.join(trt_dir, f"dfine_trt_batch{batch_size}.plan")

    # Override trt_path if custom file specified
    if trt_file:
        trt_path = trt_file

    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(trt_dir, exist_ok=True)

    gpu_compute_ms = None

    if benchmark_only:
        # Skip ONNX export and TensorRT build, just load existing plan
        if not os.path.exists(trt_path):
            print(f"ERROR: TensorRT plan not found: {trt_path}")
            return None
        print("=" * 60)
        print("BENCHMARK ONLY MODE - Loading existing TensorRT engine")
        print("=" * 60)
        print(f"Using existing plan: {trt_path}")
    else:
        # Step 1: Export ONNX with only final outputs
        if not os.path.exists(onnx_path):
            print("=" * 60)
            print("Step 1: Export ONNX with final outputs only")
            print("=" * 60)

            model, image_processor, device = load_model(config)
            inputs, _ = prepare_inputs(
                config.test_image_path,
                image_processor,
                device,
                "float32",  # Export in FP32, TensorRT will handle FP16
            )

            export_onnx_final_outputs_only(model, inputs, onnx_path, batch_size)
        else:
            print(f"Using existing ONNX: {onnx_path}")

        # Step 2: Build TensorRT engine
        print("\n" + "=" * 60)
        print("Step 2: Build TensorRT engine")
        print("=" * 60)

        trt_path, gpu_compute_ms = build_tensorrt_from_onnx(onnx_path, trt_path, batch_size, precision != "fp32")

    if trt_path is None:
        print("TensorRT build failed!")
        return None

    # Step 3: Benchmark
    print("\n" + "=" * 60)
    print("Step 3: Benchmark TensorRT engine")
    print("=" * 60)

    if use_docker:
        # Use Docker for benchmark (same TRT version as build)
        latency_results = run_tensorrt_benchmark_docker(trt_path)
    else:
        # Use local TensorRT Python bindings
        latency_results = run_tensorrt_benchmark(trt_path, batch_size)

    if latency_results is None and gpu_compute_ms:
        # Use trtexec results if Python TRT bindings not available
        latency_results = {
            "mean_ms": gpu_compute_ms,
            "std_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
            "source": "trtexec"
        }

    # Step 4: Accuracy evaluation
    accuracy_results = {"mAP_50": None, "mAP_50_95": None}
    if run_accuracy:
        if use_docker:
            print("\nNote: Accuracy evaluation skipped when using --use-docker")
            print("(Docker-built plans use TRT 8.6, but local TRT is 10.x)")
        else:
            accuracy_results = run_tensorrt_accuracy_evaluation(trt_path, config, batch_size)

    # Calculate throughput
    if latency_results and latency_results.get("mean_ms"):
        throughput = batch_size * 1000 / latency_results["mean_ms"]
    else:
        throughput = None

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Experiment: 07b_tensorrt_trtexec")
    print(f"Batch size: {batch_size}")
    print(f"Precision: {precision.upper()}")
    print(f"Plan file: {trt_path}")
    if latency_results:
        print(f"Latency (mean): {latency_results['mean_ms']:.2f} ms")
    if throughput:
        print(f"Throughput: {throughput:.2f} images/sec")
    if accuracy_results["mAP_50"] is not None:
        print(f"mAP@50: {accuracy_results['mAP_50']:.4f}")
        print(f"mAP@50:95: {accuracy_results['mAP_50_95']:.4f}")

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, "output", "07b_tensorrt_trtexec")
    os.makedirs(output_dir, exist_ok=True)

    results_data = {
        "config": {
            "experiment_name": "07b_tensorrt_trtexec",
            "batch_size": batch_size,
            "precision": precision.lower(),
            "trt_path": trt_path,
        },
        "latency": latency_results,
        "throughput": {"images_per_sec": throughput},
        "accuracy": accuracy_results,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"],
                        help="Precision mode: fp32, fp16, or int8")
    parser.add_argument("--no-accuracy", action="store_true", help="Skip accuracy evaluation")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip ONNX export and TensorRT build, just benchmark existing plan")
    parser.add_argument("--trt-file", type=str, default=None,
                        help="Path to existing TensorRT plan file (use with --benchmark-only)")
    parser.add_argument("--use-docker", action="store_true",
                        help="Use Docker for both build and benchmark (ensures TRT version consistency)")
    args = parser.parse_args()

    main(args.batch_size, args.precision, not args.no_accuracy,
         args.benchmark_only, args.trt_file, args.use_docker)
