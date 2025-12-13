before applying something to claude.md, tell me some change FIRST

# D-Fine Model Inference Optimization

## Goal
Maximize requests per second (throughput) by reducing inference latency for D-Fine model.

## Hardware
- GPU: Tesla T4 (15GB VRAM)

## Model Configuration
- Model: `ustc-community/dfine_x_coco`
- Architecture: `DFineForObjectDetection`
- Backbone: HGNet_v2
- d_model: 256
- Encoder layers: 1, Decoder layers: 6
- Number of queries: 300
- Classes: 80 COCO categories
- Default dtype: `float32`
- Config path: `~/.cache/huggingface/hub/models--ustc-community--dfine_x_coco/snapshots/ea4f6be7350bbe3c199ec6febc74168346cb5a68/`

### Preprocessor Settings
- Image processor: RTDetrImageProcessor
- Input size: 640x640
- Resample: 2 (BILINEAR)
- Rescale factor: 0.00392156862745098 (1/255)
- Image mean: [0.485, 0.456, 0.406]
- Image std: [0.229, 0.224, 0.225]

## Project Structure
- `image/` - Local images for inference
  - `000000039769.jpg` - Sample COCO dataset image
- `scripts/` - Python scripts for inference

## Usage
Images should be stored locally in the `image/` folder to avoid network latency during inference.

## Accuracy Measurement
- Validation set: COCO val2017 subset (100 images)
- Metrics: mAP@50, mAP@50:95
- Minimum acceptable: 95% of baseline mAP

## Measurement Methods

### PyTorch Direct (Scripts 03-11)
- Latency: `torch.cuda.synchronize()` + `time.perf_counter()`
- Warmup: 10 iterations
- Benchmark: 100 iterations
- Metrics: mean, std, p50, p95, p99, RPS

### Batching (Script 10)
- Same as PyTorch Direct
- Test batch sizes: 1, 2, 4, 8
- RPS = batch_size * 1000 / latency_ms

### Triton Server (Script 11)
- Throughput: `perf_analyzer` tool
- Accuracy: Custom Python client with same COCO evaluation


## Optimization Plan

### Phase 0: Validation
1. **Visualization** - Visualize object detection results with bounding boxes to verify model works correctly

### Phase 1: Baseline & Quick Wins
2. **Baseline Measurement** - Measure GPU inference latency (CUDA)
3. **Reduce num_queries** - Reduce from 300 to 100-150 at runtime (fewer object candidates = faster decoder)
4. **Half Precision (FP16)** - Use `torch.float16` for faster computation on T4
5. **torch.compile()** - Use PyTorch 2.0+ compilation for graph optimization

### Phase 2: Runtime Optimization
6. **TensorRT Conversion** - Export to TensorRT for maximum GPU performance
7. **ONNX Runtime** - Alternative: export to ONNX with GPU execution provider
8. **INT8 Quantization** - Quantize model to INT8 for faster inference (requires calibration)

### Phase 3: Throughput Optimization (maximize requests/sec)
9. **Batching** - Process multiple images in batches
10. **Triton Inference Server** - Dynamic batching, concurrent execution for high throughput


### Expected Performance Gains
| Optimization | Expected Speedup | Accuracy Impact |
|--------------|------------------|-----------------|
| Baseline (CUDA) | 1x | None |
| num_queries 100 | ~1.2x | -1-3% mAP |
| FP16 | 1.5-2x | <1% mAP |
| torch.compile | 1.2-1.5x | None |
| TensorRT | 2-4x | <1% mAP |
| ONNX Runtime | 1.5-2x | <1% mAP |
| INT8 Quantization | 2-4x | -1-5% mAP |
| Batching | 2-4x throughput | None |
| Triton Server | 3-10x throughput | None |

### Scripts

**Metrics per script:** All scripts (03-11) output Latency (ms), RPS, mAP@50, mAP@50:95
- Exception: `02_visualize.py` - Visual validation only (no metrics)

- `02_visualize.py` - Visualize object detection with bounding boxes
- `03_baseline_fp32.py` - Baseline (CUDA + FP32)
- `04_num_queries.py` - Test num_queries (300, 150, 100) **(Not improved - skip)**
- `05_fp16.py` - FP16 optimization
- `06_torch_compile.py` - PyTorch compilation
- `07_tensorrt_torch.py` - TensorRT via torch_tensorrt (deprecated, accuracy issues)
- `08_tensorrt_native.py` - Native TensorRT using trtexec **(recommended)**
- `09_onnx_runtime.py` - ONNX Runtime optimization
- `10_batching.py` - Batch inference benchmark
- `11_triton_server.py` - Triton Inference Server benchmark

## Benchmark Results Summary

### Single Image Latency (batch=1)
| Optimization | Latency (ms) | RPS | mAP@50 | Notes |
|--------------|--------------|-----|--------|-------|
| Baseline FP32 | 83.96 | 11.91 | 0.684 | Script 03 |
| FP16 | 59.54 | 16.80 | 0.684 | Script 05 |
| torch.compile (reduce-overhead) | 60.71 | 16.47 | 0.684 | Script 06 |
| TensorRT via torch.compile | 87.48 | 11.43 | 0.463 | Script 07 - deprecated |
| ONNX Runtime | 65.94 | 15.17 | 0.684 | Script 09 |

### Batched Inference (FP16)
| Batch Size | Latency (ms) | Images/sec | Notes |
|------------|--------------|------------|-------|
| 1 | 59.79 | 16.73 | Script 10 |
| 2 | 71.25 | 28.07 | |
| 4 | 103.91 | 38.49 | Good balance |
| 8 | 165.41 | 48.37 | |
| 16 | 389.24 | 41.10 | Diminishing returns |

### Triton Inference Server (Script 11)

#### ONNX Runtime Backend
- Latency: 729.99ms (batch=4)
- Throughput: 5.48 images/sec
- Note: High latency due to HTTP overhead + ONNX Runtime slower than native PyTorch

#### TensorRT Engine (built locally via trtexec)
- GPU Compute: **51.24ms** (batch=4)
- Throughput: **~77 images/sec** (best achieved)
- Engine file: `output/tensorrt/dfine_trt_batch4.plan`

#### Triton + TensorRT Challenges
The D-Fine model's ONNX export produces 16 output tensors (intermediate outputs from encoder/decoder), but Triton config expects only 2 (`logits`, `pred_boxes`). This causes model loading to fail.

**Workarounds:**
1. Use ONNX with TensorRT Execution Provider (works, slower)
2. Re-export ONNX with only final outputs (requires model modification)
3. Use TensorRT engine directly without Triton (best performance)


### Official D-FINE Repository Comparison

Compared our HuggingFace-based approach with official D-FINE tools from https://github.com/Peterande/D-FINE

#### Official Tools Used
- `tools/deployment/export_onnx.py` - ONNX export with `model.deploy()` mode
- `tools/benchmark/trt_benchmark.py` - TensorRT benchmarking

#### Key Differences
| Aspect | Our Approach (HuggingFace) | Official D-FINE |
|--------|---------------------------|-----------------|
| Model source | `ustc-community/dfine_x_coco` | Official weights |
| Deploy mode | Not available | `model.deploy()` optimizes decoder |
| ONNX outputs | 16 → trimmed to 2 | 3 (includes postprocessor) |
| Postprocessor | Separate Python | Integrated in ONNX |

#### TensorRT Latency Comparison
| Method | Batch | GPU Latency | Per-Image | Throughput |
|--------|-------|-------------|-----------|------------|
| Our Approach | 1 | 14.04 ms | 14.04 ms | 70.9 img/s |
| Our Approach | 4 | 48.59 ms | 12.15 ms | 81.0 img/s |
| Official Export | 1 | 14.20 ms | 14.20 ms | 70.4 img/s |

**Conclusion:** With batch=1, our HuggingFace TensorRT engine (14.04ms) matches the official D-FINE export (14.20ms). Batch=4 provides ~14% higher throughput (81 img/s vs 71 img/s) at the cost of higher per-request latency.

## Production Recommendations

### For Maximum Throughput (Direct Serving)
- Use **FP16 + batch=4-8** with native PyTorch
- Achieves 38-48 images/sec with simple setup

### For Triton Deployment
- **Option 1 (Simple):** Use ONNX Runtime backend with TensorRT EP
- **Option 2 (Best):** Fix ONNX export, use pre-built TensorRT engine
- **Option 3 (Alternative):** Skip Triton, use TensorRT directly (~51ms latency)
