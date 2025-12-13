import torch
import time
import os
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
from transformers.image_utils import load_image
from transformers import DFineForObjectDetection, AutoImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw, ImageFont


# Project root directory
PROJECT_ROOT = "/home/ubuntu/lab/17-d-fine-model-inference-optimization"

# Test images for visualization (easy, medium, hard)
TEST_IMAGES = {
    "easy": f"{PROJECT_ROOT}/data/coco/val2017/000000041888.jpg",      # 3 objects
    "medium": f"{PROJECT_ROOT}/data/coco/val2017/000000186980.jpg",    # 14 objects, 10 categories
    "hard": f"{PROJECT_ROOT}/data/coco/val2017/000000226147.jpg",      # 21 objects, 13 categories
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Benchmark settings
    num_warmup: int = 10
    num_iterations: int = 100
    validation_images: int = 100
    confidence_threshold: float = 0.5

    # Paths (absolute)
    coco_val_path: str = f"{PROJECT_ROOT}/data/coco/val2017"
    coco_ann_path: str = f"{PROJECT_ROOT}/data/coco/annotations/instances_val2017.json"
    test_image_path: str = f"{PROJECT_ROOT}/image/000000039769.jpg"

    # Model settings
    model_name: str = "ustc-community/dfine_x_coco"
    device: str = "cuda"
    dtype: str = "float32"

    # Optimization-specific settings (to be overridden)
    num_queries: Optional[int] = None  # None means use default (300)
    use_compile: bool = False
    compile_mode: Optional[str] = None  # "default", "reduce-overhead", "max-autotune"
    compile_backend: Optional[str] = None  # None (inductor), "torch_tensorrt"
    use_onnx: bool = False  # Use ONNX Runtime for inference
    batch_sizes: Optional[list] = None  # Test multiple batch sizes (e.g., [1, 2, 4, 8])

    # Output
    output_dir: str = f"{PROJECT_ROOT}/output"
    experiment_name: str = "baseline"

    def get_output_path(self) -> str:
        return os.path.join(self.output_dir, self.experiment_name)


def print_config(config: BenchmarkConfig) -> None:
    """Print test configuration at the start."""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {config.experiment_name}")
    print(f"{'='*50}")
    print(f"Device: {config.device}")
    print(f"Dtype: {config.dtype}")
    if config.num_queries is not None:
        print(f"Num queries: {config.num_queries}")
    if config.use_compile:
        backend = config.compile_backend or "inductor"
        print(f"torch.compile: {config.compile_mode or 'default'} (backend: {backend})")
    if config.use_onnx:
        print("Runtime: ONNX Runtime (GPU)")
    if config.batch_sizes:
        print(f"Batch sizes: {config.batch_sizes}")
    print(f"Warmup: {config.num_warmup} iterations")
    print(f"Benchmark: {config.num_iterations} iterations")
    print(f"Validation images: {config.validation_images}")
    print(f"{'='*50}\n")


def load_model(
    config: BenchmarkConfig,
) -> Tuple[DFineForObjectDetection, AutoImageProcessor, torch.device]:
    """Load model and processor with optional optimizations."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    image_processor = AutoImageProcessor.from_pretrained(config.model_name, use_fast=True)
    model = DFineForObjectDetection.from_pretrained(config.model_name)

    # Apply num_queries modification if specified
    if config.num_queries is not None:
        print(f"Setting num_queries to {config.num_queries}")
        model.config.num_queries = config.num_queries

    # Move to device
    model = model.to(device)

    # Apply dtype
    if config.dtype == "float16":
        print("Converting model to FP16")
        model = model.half()

    # Apply torch.compile if specified
    if config.use_compile:
        backend = config.compile_backend or "inductor"
        print(f"Compiling model with mode: {config.compile_mode or 'default'} (backend: {backend})")
        if config.compile_backend == "torch_tensorrt":
            import torch_tensorrt
            model = torch.compile(
                model,
                backend="torch_tensorrt",
                options={
                    "truncate_long_and_double": True,
                    "precision": torch.float16 if config.dtype == "float16" else torch.float32,
                },
            )
        else:
            model = torch.compile(model, mode=config.compile_mode)

    model.eval()

    return model, image_processor, device


def prepare_inputs(
    image_path: str,
    image_processor: AutoImageProcessor,
    device: torch.device,
    dtype: str = "float32",
) -> Tuple[Dict[str, torch.Tensor], Any]:
    """Load image and prepare inputs for the model."""
    image = load_image(image_path)
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Convert to FP16 if needed
    if dtype == "float16":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    return inputs, image


def export_to_onnx(model, inputs, onnx_path):
    """Export PyTorch model to ONNX format."""
    print(f"Exporting model to ONNX: {onnx_path}")

    pixel_values = inputs["pixel_values"]

    # Use legacy exporter (dynamo=False) for better compatibility
    torch.onnx.export(
        model,
        (pixel_values,),
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"ONNX model saved to: {onnx_path}")


def create_onnx_session(onnx_path: str):
    """Create ONNX Runtime inference session with GPU provider."""
    import onnxruntime as ort

    print(f"\nONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if "CUDAExecutionProvider" in providers:
        session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("Using CUDAExecutionProvider")
    else:
        session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        print("WARNING: CUDA not available, using CPU")

    return session


def run_latency_benchmark(
    model: DFineForObjectDetection,
    inputs: Dict[str, torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Run latency benchmark and return statistics.

    Note: GPU stats should be monitored externally with gpu_monitor.sh
    """
    device = next(model.parameters()).device

    # Warmup
    print(f"\nWarmup ({config.num_warmup} iterations)...")
    for _ in range(config.num_warmup):
        with torch.no_grad():
            _ = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking latency ({config.num_iterations} iterations)...")
    latencies = []

    for _ in range(config.num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def run_onnx_latency_benchmark(
    session,
    inputs: Dict[str, torch.Tensor],
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Run latency benchmark for ONNX Runtime session."""
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: inputs["pixel_values"].cpu().numpy()}

    # Warmup
    print(f"\nWarmup ({config.num_warmup} iterations)...")
    for _ in range(config.num_warmup):
        _ = session.run(None, ort_inputs)

    # Benchmark
    print(f"Benchmarking latency ({config.num_iterations} iterations)...")
    latencies = []
    for _ in range(config.num_iterations):
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
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


def run_accuracy_evaluation(
    model: DFineForObjectDetection,
    image_processor: AutoImageProcessor,
    config: BenchmarkConfig,
) -> Dict[str, Optional[float]]:
    """Run mAP evaluation on COCO validation set."""
    device = next(model.parameters()).device

    print(f"\n{'='*50}")
    print(f"ACCURACY EVALUATION ({config.validation_images} images)")
    print(f"{'='*50}")

    # Check if COCO dataset exists
    if not (os.path.exists(config.coco_ann_path) and os.path.exists(config.coco_val_path)):
        print(f"COCO dataset not found at {config.coco_val_path}")
        print("Skipping mAP evaluation. Please download COCO val2017 dataset.")
        return {"mAP_50": None, "mAP_50_95": None}

    # Suppress COCO loading messages
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_gt = COCO(config.coco_ann_path)
    sys.stdout = old_stdout

    # Get image ids (limit to validation_images)
    img_ids = coco_gt.getImgIds()[:config.validation_images]

    # COCO category mapping
    coco_categories = coco_gt.loadCats(coco_gt.getCatIds())
    coco_cat_ids = [cat['id'] for cat in coco_categories]

    results = []

    print(f"Running inference on {len(img_ids)} images...")
    for idx, img_id in enumerate(img_ids):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(img_ids)}")

        # Load image
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(config.coco_val_path, img_info['file_name'])
        image = load_image(img_path)

        # Prepare inputs
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Convert to FP16 if needed
        if config.dtype == "float16":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = [(image.height, image.width)]
        detections = image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=config.confidence_threshold
        )[0]

        # Convert to COCO format
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            coco_cat_id = coco_cat_ids[label.item()]
            results.append({
                "image_id": img_id,
                "category_id": coco_cat_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score.item()
            })

    # Evaluate
    if results:
        # Suppress COCO evaluation messages
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

        return {
            "mAP_50": map_50,
            "mAP_50_95": map_50_95,
        }
    else:
        print("No detections found!")
        return {"mAP_50": 0.0, "mAP_50_95": 0.0}


def print_summary(
    latency_results: Dict[str, float],
    accuracy_results: Dict[str, Optional[float]],
    config: BenchmarkConfig,
) -> None:
    """Print formatted benchmark summary."""
    rps = 1000 / latency_results["mean_ms"]

    print(f"\n{'='*50}")
    print(f"LATENCY RESULTS ({config.dtype.upper()})")
    print(f"{'='*50}")
    print(f"Mean latency: {latency_results['mean_ms']:.2f} ms")
    print(f"Std latency: {latency_results['std_ms']:.2f} ms")
    print(f"P50 latency: {latency_results['p50_ms']:.2f} ms")
    print(f"P95 latency: {latency_results['p95_ms']:.2f} ms")
    print(f"P99 latency: {latency_results['p99_ms']:.2f} ms")
    print(f"Throughput (RPS): {rps:.2f}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Latency (mean): {latency_results['mean_ms']:.2f} ms")
    print(f"RPS: {rps:.2f}")
    if accuracy_results["mAP_50"] is not None:
        print(f"mAP@50: {accuracy_results['mAP_50']:.4f}")
        print(f"mAP@50:95: {accuracy_results['mAP_50_95']:.4f}")


def save_results(
    latency_results: Dict[str, float],
    accuracy_results: Dict[str, Optional[float]],
    config: BenchmarkConfig,
    extra_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Save benchmark results to JSON file."""
    output_path = config.get_output_path()
    os.makedirs(output_path, exist_ok=True)

    rps = 1000 / latency_results["mean_ms"]

    results_dict = {
        "config": {
            "experiment_name": config.experiment_name,
            "device": config.device,
            "dtype": config.dtype,
            "num_iterations": config.num_iterations,
            "validation_images": config.validation_images,
            "confidence_threshold": config.confidence_threshold,
            "num_queries": config.num_queries,
            "use_compile": config.use_compile,
            "compile_mode": config.compile_mode,
        },
        "latency": latency_results,
        "throughput": {
            "rps": rps,
        },
        "accuracy": accuracy_results,
    }

    if extra_info:
        results_dict["extra"] = extra_info

    output_file = os.path.join(output_path, "results.json")
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file


def save_visualizations(
    model: DFineForObjectDetection,
    image_processor: AutoImageProcessor,
    config: BenchmarkConfig,
) -> None:
    """Save detection visualizations for easy, medium, hard test images."""
    device = next(model.parameters()).device
    output_path = config.get_output_path()
    os.makedirs(output_path, exist_ok=True)

    # Color palette for different classes
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FF8000", "#8000FF", "#0080FF", "#FF0080", "#80FF00", "#00FF80"
    ]

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    print(f"\n{'='*50}")
    print("SAVING VISUALIZATIONS")
    print(f"{'='*50}")

    for difficulty, image_path in TEST_IMAGES.items():
        if not os.path.exists(image_path):
            print(f"  {difficulty}: Image not found, skipping")
            continue

        # Load image
        image = load_image(image_path)

        # Prepare inputs
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if config.dtype == "float16":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=[(image.height, image.width)],
            threshold=config.confidence_threshold
        )[0]

        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        num_detections = len(results["scores"])

        for idx, (score, label_id, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            score_val = score.item()
            label = label_id.item()
            box_coords = [round(i, 2) for i in box.tolist()]

            label_name = model.config.id2label[label]
            color = colors[label % len(colors)]

            # Draw bounding box
            x1, y1, x2, y2 = box_coords
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background and text
            text = f"{label_name}: {score_val:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1), text, fill="white", font=font)

        # Save image
        output_file = os.path.join(output_path, f"detection_{difficulty}.jpg")
        image.save(output_file)
        print(f"  {difficulty}: {num_detections} detections -> {output_file}")


def run_batch_latency_benchmark(
    model: DFineForObjectDetection,
    inputs: Dict[str, torch.Tensor],
    batch_size: int,
    config: BenchmarkConfig,
) -> Dict[str, float]:
    """Run latency benchmark for a specific batch size."""
    device = next(model.parameters()).device
    pixel_values = inputs["pixel_values"]
    batched_input = pixel_values.repeat(batch_size, 1, 1, 1)

    # Warmup
    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model(batched_input)
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(config.num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batched_input)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    mean_latency = float(np.mean(latencies))
    throughput = batch_size * 1000 / mean_latency

    return {
        "batch_size": batch_size,
        "mean_ms": mean_latency,
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_ips": throughput,
        "latency_per_image_ms": mean_latency / batch_size,
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run complete benchmark with given configuration."""
    # Print test configuration
    print_config(config)

    # Load model
    model, image_processor, device = load_model(config)

    # Prepare inputs for latency test
    inputs, _ = prepare_inputs(
        config.test_image_path,
        image_processor,
        device,
        config.dtype,
    )

    # Batching mode - test multiple batch sizes
    if config.batch_sizes:
        print("\n" + "=" * 60)
        print("BATCH SIZE COMPARISON")
        print("=" * 60)

        batch_results = []
        for batch_size in config.batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            try:
                result = run_batch_latency_benchmark(model, inputs, batch_size, config)
                batch_results.append(result)
                print(f"  Mean latency: {result['mean_ms']:.2f} ms")
                print(f"  Throughput: {result['throughput_ips']:.2f} images/sec")
                print(f"  Latency per image: {result['latency_per_image_ms']:.2f} ms")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at batch size {batch_size}, skipping larger sizes")
                    torch.cuda.empty_cache()
                    break
                raise

        # Find optimal batch size
        if batch_results:
            best = max(batch_results, key=lambda x: x["throughput_ips"])
            print("\n" + "=" * 60)
            print("OPTIMAL BATCH SIZE")
            print("=" * 60)
            print(f"Best batch size: {best['batch_size']}")
            print(f"Best throughput: {best['throughput_ips']:.2f} images/sec")
            print(f"Mean latency: {best['mean_ms']:.2f} ms")

        # Save results
        output_path = config.get_output_path()
        os.makedirs(output_path, exist_ok=True)
        results_data = {
            "experiment": config.experiment_name,
            "batch_results": batch_results,
            "optimal_batch_size": best["batch_size"] if batch_results else None,
        }
        results_path = os.path.join(output_path, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        return results_data

    # ONNX Runtime mode
    if config.use_onnx:
        # Export to ONNX
        onnx_dir = os.path.join(PROJECT_ROOT, "output", "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        onnx_path = os.path.join(onnx_dir, "dfine_x_coco.onnx")

        if not os.path.exists(onnx_path):
            export_to_onnx(model, inputs, onnx_path)
        else:
            print(f"Using existing ONNX model: {onnx_path}")

        # Create ONNX session and run latency benchmark
        session = create_onnx_session(onnx_path)
        latency_results = run_onnx_latency_benchmark(session, inputs, config)

        # Use PyTorch model for accuracy (ONNX output format differs)
        print("\nNote: Using PyTorch model for accuracy evaluation")
    else:
        # Run latency benchmark with PyTorch model
        latency_results = run_latency_benchmark(model, inputs, config)

    # Run accuracy evaluation (always with PyTorch model)
    accuracy_results = run_accuracy_evaluation(model, image_processor, config)

    # Print summary
    print_summary(latency_results, accuracy_results, config)

    # Save results
    save_results(latency_results, accuracy_results, config)

    # Save visualizations
    save_visualizations(model, image_processor, config)

    return {
        "latency": latency_results,
        "accuracy": accuracy_results,
        "rps": 1000 / latency_results["mean_ms"],
    }
