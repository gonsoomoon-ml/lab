# 상세 정보

## 목차

1. [환경 설정](#환경-설정)
2. [검증 데이터셋](#검증-데이터셋)
3. [전체 스크립트 목록](#전체-스크립트-목록)
4. [TensorRT 스크립트 상세](#08-tensorrt-스크립트-상세)
5. [상세 벤치마크 결과](#상세-벤치마크-결과)
6. [Triton 서버 사용](#triton-서버-사용)
7. [문제 해결](#문제-해결)
8. [참고 자료](#참고-자료)

---

## 환경 설정

### 요구 사항
- Python 3.11+
- CUDA 12.x
- Docker (nvidia-docker 런타임) - TensorRT 빌드/Triton용

### 주요 패키지 버전

| 패키지 | 버전 | 용도 |
|-------|------|-----|
| Python | 3.11.14 | 런타임 |
| CUDA | 12.9 | GPU 컴퓨팅 |
| NVIDIA Driver | 580.95 | GPU 드라이버 |
| PyTorch | 2.9.1+cu128 | 딥러닝 프레임워크 |
| TensorRT | 10.13.3 | 추론 최적화 |
| Transformers | 4.57.3 | D-Fine 모델 |
| ONNX | 1.19.1 | 모델 변환 |
| ONNX Runtime | 1.23.2 | ONNX 추론 |
| Triton Client | 2.42+ | 추론 서버 클라이언트 |

### 빌드/추론 환경

| 작업 | 스크립트 | 환경 | Docker 이미지 | TRT 버전 |
|-----|---------|-----|--------------|---------|
| FP16/INT8 빌드 | `08_tensorrt_native.py` | Docker | `nvcr.io/nvidia/tensorrt:24.01-py3` | 8.6 |
| FP16/INT8 빌드 | `08_tensorrt_native.py --use-docker` | Docker | `nvcr.io/nvidia/tensorrt:24.01-py3` | 8.6 |
| FP16/INT8 추론 | `08_tensorrt_native.py` (기본) | 로컬 | - | 10.x |
| FP16/INT8 추론 | `08_tensorrt_native.py --use-docker` | Docker | `nvcr.io/nvidia/tensorrt:24.01-py3` | 8.6 |
| Triton 서버 | `11_triton_server.py` | Docker | `nvcr.io/nvidia/tritonserver:25.10-py3` | 10.x |

> **⚠️ 주의**: `08_tensorrt_native.py`는 Docker TRT 8.6으로 빌드하지만, 기본 추론은 로컬 TRT 10.x를 사용합니다.
> 버전 불일치로 플랜 로드 실패 시 `--use-docker` 옵션을 사용하세요.

### 설치

```bash
# 최초 환경 설정 (UV 설치 + 의존성 + Jupyter 커널)
cd setup && ./create_env.sh dfine && cd ..

# 테스트 실행
uv run python scripts/01_quick_start.py
```

> **참고**: `create_env.sh`는 UV 설치, 의존성 동기화, Jupyter 커널 등록을 자동으로 수행합니다.

---

## 검증 데이터셋

### COCO val2017

정확도(mAP) 측정을 위해 COCO val2017 데이터셋을 사용합니다.

| 항목 | 값 |
|-----|---|
| 데이터셋 | COCO val2017 |
| 이미지 수 | 100장 (전체 5,000장 중 샘플링) |
| 경로 | `data/coco/val2017/` |
| 어노테이션 | `data/coco/annotations/instances_val2017.json` |

### 평가 지표

| 지표 | 설명 |
|-----|------|
| mAP@50 | IoU 임계값 0.5에서의 평균 정밀도 |
| mAP@50:95 | IoU 0.5~0.95 (0.05 간격) 평균 |

### 데이터 다운로드

```bash
# COCO val2017 이미지
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/

# 어노테이션
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/coco/
```

---

## 전체 스크립트 목록

> **실행 방법**: `uv run python scripts/<스크립트명>`

| 스크립트 | 설명 | 비고 |
|---------|------|-----|
| `01_quick_start.py` | 빠른 시작 데모 | |
| `02_visualize.py` | 탐지 결과 시각화 | 정확도 측정 없음 |
| `03_baseline_fp32.py` | 기준 성능 측정 (FP32) | |
| `04_num_queries.py` | 쿼리 수 테스트 | 성능 향상 미미 |
| `05_fp16.py` | FP16 최적화 | 1.3x 속도 향상 |
| `06_torch_compile.py` | torch.compile 최적화 | 1.2x 속도 향상 |
| `07_tensorrt_torch.py` | torch_tensorrt | ⚠️ 권장X (정확도 손실) |
| `08_tensorrt_native.py` | 네이티브 TensorRT | ✅ **권장** (5.5~8.9x) |
| `09_onnx_runtime.py` | ONNX Runtime | 1.1x 속도 향상 |
| `10_batching.py` | 배치 크기별 성능 | |
| `11_triton_server.py` | Triton 서버 설정 | |

---

## 08 TensorRT 스크립트 상세

### 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--batch-size` | 배치 크기 | 1 |
| `--precision` | 정밀도 (fp32, fp16, int8) | fp16 |
| `--benchmark-only` | 기존 플랜으로 벤치마크만 | - |
| `--trt-file` | TensorRT 플랜 파일 경로 | - |
| `--use-docker` | Docker로 빌드/벤치마크 | - |
| `--no-accuracy` | 정확도 평가 건너뛰기 | - |

### 사용 예시

```bash
# 1. 플랜 파일 생성 (최초 1회)
uv run python scripts/08_tensorrt_native.py --precision fp16       # FP16
uv run python scripts/08_tensorrt_native.py --precision int8       # INT8

# 2. 기존 플랜으로 벤치마크
uv run python scripts/08_tensorrt_native.py \
    --benchmark-only \
    --trt-file output/tensorrt/dfine_trt_batch1_local.plan \
    --precision fp16

uv run python scripts/08_tensorrt_native.py \
    --benchmark-only \
    --trt-file output/tensorrt/dfine_trt_batch1_int8_calibrated.plan \
    --precision int8

# Docker로 빌드/벤치마크 (TRT 버전 일치 보장)
uv run python scripts/08_tensorrt_native.py --precision fp16 --use-docker
```


---

## 상세 벤치마크 결과

### 단일 이미지 (batch=1)

| 최적화 방식 | 지연시간 (ms) | 처리량 (img/s) | mAP@50 | mAP@50:95 | 속도 향상 |
|------------|--------------|----------------|--------|-----------|----------|
| Baseline (FP32) | 71.20 | 14.04 | 0.684 | 0.572 | 1x |
| FP16 | 56.01 | 17.85 | 0.682 | 0.574 | 1.3x |
| torch.compile | 59.87 | 16.70 | 0.684 | 0.572 | 1.2x |
| ONNX Runtime | 64.99 | 15.39 | 0.684 | 0.572 | 1.1x |
| **TensorRT FP16** | **12.98** | **77.06** | **0.682** | **0.565** | **5.5x** |
| **TensorRT INT8** | **8.04** | **124.38** | **0.658** | **0.546** | **8.9x** |

### 배치 처리 (FP16)

| 배치 크기 | 지연시간 (ms) | 처리량 (img/s) |
|----------|--------------|----------------|
| 1 | 55.04 | 18.17 |
| 2 | 61.60 | 32.46 |
| 4 | 103.03 | 38.82 |
| 8 | 197.22 | 40.56 |
| 16 | 387.87 | 41.25 |

### INT8 캘리브레이션 비교

| 캘리브레이션 | mAP@50 | 정확도 손실 |
|-------------|--------|------------|
| 랜덤 데이터 | 0.631 | -7.5% |
| COCO 100장 | 0.658 | -3.4% |

---

## Triton 서버 사용

```bash
# 모델 저장소 설정
uv run python scripts/11_triton_server.py \
    --trt-file output/tensorrt/dfine_trt_batch1_int8_calibrated.plan \
    --precision int8

# Triton 서버 시작
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/output/triton_model_repository:/models \
    nvcr.io/nvidia/tritonserver:25.10-py3 \
    tritonserver --model-repository=/models
```

---

## 문제 해결

### TensorRT 플랜 로드 실패

```
Error Code 1: Internal Error (Failed due to an old deserialization call)
```

**원인**: Docker TRT 8.6으로 빌드한 플랜을 로컬 TRT 10.x에서 로드 시 발생

| 환경 | TRT 버전 | 빌드 | 로컬 로드 |
|-----|---------|------|----------|
| Docker (24.01-py3) | 8.6 | O | X |
| 로컬 | 10.13 | O | O |

**해결**:
1. **권장**: 로컬에서 플랜 빌드 (`--precision fp16` 또는 `--precision int8`)
2. **대안**: `--use-docker` 플래그로 Docker 내에서 빌드 및 벤치마크

### CUDA Out of Memory

**원인**: 배치 크기가 너무 크거나 GPU 메모리 부족

**해결**: 배치 크기 줄이기
```bash
uv run python scripts/08_tensorrt_native.py --batch-size 1
```

### torch_tensorrt 정확도 손실

**원인**: `07_tensorrt_torch.py`는 D-Fine 모델과 호환성 문제 (mAP 0.46)

**해결**: 네이티브 TensorRT 사용
```bash
uv run python scripts/08_tensorrt_native.py --precision fp16
```

---

## 참고 자료

- [D-FINE 공식 저장소](https://github.com/Peterande/D-FINE)
- [HuggingFace D-Fine 모델](https://huggingface.co/ustc-community/dfine_x_coco)
- [TensorRT 문서](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
