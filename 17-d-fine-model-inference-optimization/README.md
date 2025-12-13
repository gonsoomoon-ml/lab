# D-Fine 모델 추론 최적화

D-Fine 객체 탐지 모델의 추론 성능을 최적화하는 프로젝트입니다.

## 환경

- **인스턴스**: AWS EC2 g4dn.xlarge (Tesla T4, 16GB VRAM)
- **모델**: `ustc-community/dfine_x_coco` (640x640, 80 클래스)
- **검증 데이터**: COCO val2017 (100장, mAP@50/mAP@50:95 측정)

## 빠른 시작

```bash
# 최초 환경 설정 (UV 설치 + 의존성 + Jupyter 커널)
cd setup && ./create_env.sh dfine && cd ..

# 테스트 실행
uv run python scripts/01_quick_start.py
```

## 벤치마크 결과 요약

| 최적화 방식 | 지연시간 | 처리량 | mAP@50 | 속도 향상 |
|------------|---------|--------|--------|----------|
| Baseline (FP32) | 71.20 ms | 14 img/s | 0.684 | 1x |
| FP16 | 56.01 ms | 18 img/s | 0.682 | 1.3x |
| **TensorRT FP16** | **12.98 ms** | **77 img/s** | **0.682** | **5.5x** |
| **TensorRT INT8** | **8.04 ms** | **124 img/s** | **0.658** | **8.9x** |

## 주요 스크립트

| 스크립트 | 설명 |
|---------|------|
| `03_baseline_fp32.py` | 기준 성능 측정 |
| `05_fp16.py` | FP16 최적화 |
| `08_tensorrt_native.py` | TensorRT 빌드/벤치마크 **(권장)** |
| `10_batching.py` | 배치 크기별 성능 |
| `11_triton_server.py` | Triton 서버 설정 |

## TensorRT 사용법

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
```

## 프로젝트 구조

```
scripts/
├── 01_quick_start.py       # 빠른 시작
├── 02_visualize.py         # 시각화
├── 03_baseline_fp32.py     # 기준 측정
├── 05_fp16.py              # FP16
├── 08_tensorrt_native.py   # TensorRT (권장)
├── 09_onnx_runtime.py      # ONNX Runtime
├── 10_batching.py          # 배치 처리
└── 11_triton_server.py     # Triton 서버
```

## 상세 문서

- [detail_info.md](detail_info.md) - 환경 설정, 상세 벤치마크, TensorRT 사용법, 문제 해결

## 참고 자료

- [D-FINE 공식 저장소](https://github.com/Peterande/D-FINE)
- [HuggingFace D-Fine 모델](https://huggingface.co/ustc-community/dfine_x_coco)
