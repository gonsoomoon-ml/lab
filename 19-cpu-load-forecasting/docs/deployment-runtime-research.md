# 5G UPF 런타임 환경 리서치 및 경량화 모델 배포 전략

**작성일**: 2026-03-19
**작성자**: AI Research Team
**목적**: CPU Load Forecasting 모델을 5G UPF 네트워크 장비에 배포하기 위한 런타임 환경 분석

---

## Executive Summary (주요 발견사항)

### 핵심 결론

1. **현재 모델 크기 문제**: Random Forest 모델이 **245.64MB**로 네트워크 장비에 배포하기에 너무 큼
2. **타겟 환경**: 5G UPF는 주로 **Linux 컨테이너 기반**으로 배포되며, x86-64 또는 ARM64 아키텍처 사용
3. **최적 배포 방식**: **ONNX Runtime** 기반 배포가 가장 적합 (크로스 플랫폼, 경량, 빠른 추론)
4. **최적 모델**: **XGBoost ONNX fp16 (338KB, MAPE 0.93%)** — 원본 RF 대비 99.9% 크기 감소, 정확도 거의 동일

### 액션 아이템

- [x] Random Forest 모델 경량화 (245.64MB → 0.19MB, Tree Pruning + ONNX fp16)
- [x] ONNX 포맷 변환 및 float16 양자화 적용
- [x] 추론 성능 벤치마크 수행 (1000회 반복)
- [ ] Docker 컨테이너 배포 가이드 작성

---

## 1. 오픈소스 5G UPF 구현체 분석

### 1.1 Open5GS (가장 성숙한 오픈소스)

**기본 정보:**
- **구현 언어**: C (97.5%)
- **라이선스**: Apache 2.0
- **프로젝트 성숙도**: 프로덕션 레벨

**시스템 요구사항:**
- **OS**: Ubuntu 18.04+, Debian 10+ (Linux 기반)
- **CPU 아키텍처**: x86-64, ARM64 모두 지원
- **메모리**: 2-4GB RAM (테스트 환경 기준)
- **배포 방식**: Docker, VirtualBox, 베어메탈

**주요 의존성:**
```bash
# 필수 패키지
libsctp-dev libgnutls28-dev libgcrypt-dev libssl-dev libmongoc-dev
Python3, Meson, Ninja build tools
MongoDB 8.0
```

**아키텍처 지원:**
```bash
# MongoDB 설치 시 아키텍처 명시
arch=amd64,arm64
```

### 1.2 free5GC (Go 기반 구현)

**기본 정보:**
- **구현 언어**: Go (93.5%), Shell (6.1%)
- **라이선스**: Apache 2.0
- **특징**: 경량 바이너리, 빠른 빌드

**시스템 요구사항:**
- **OS**: Linux (커널 5.0.0-23-generic 이상, 5.4+ 권장)
- **커널 모듈**: gtp5g (GTP-U 프로토콜 처리용 커널 모듈)
- **배포 방식**: 컨테이너, 베어메탈

**gtp5g 커널 모듈 특징:**
- C 언어 구현 (99.2%)
- Linux 커널 5.0+ 필수
- QoS 지원 (Session AMBR, MFBR)
- GTP-U 시퀀스 번호 지원

### 1.3 배포 환경 특성 비교

| 특성 | Open5GS | free5GC |
|------|---------|---------|
| 구현 언어 | C | Go |
| 메모리 사용량 | 중간 (C 네이티브) | 낮음 (Go 효율적) |
| 빌드 속도 | 느림 (C 컴파일) | 빠름 (Go 빌드) |
| 커널 의존성 | 낮음 | 높음 (gtp5g 모듈) |
| 배포 복잡도 | 중간 | 낮음 |
| CPU 아키텍처 | x86-64, ARM64 | Linux 지원 모든 아키텍처 |

---

## 2. 실제 배포 환경 특성

### 2.1 컨테이너화 배포 (현재 주류)

#### Kubernetes 기반 배포

**리소스 할당 패턴:**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: upf-pod
spec:
  containers:
  - name: upf
    image: upf:latest
    resources:
      requests:
        cpu: "1000m"        # 1 코어
        memory: "2Gi"       # 2GB RAM
      limits:
        cpu: "2000m"        # 최대 2 코어
        memory: "4Gi"       # 최대 4GB RAM
```

**배포 전략:**
- **CPU 할당**: 1-2 코어 기본, 최대 2-4 코어
- **메모리 할당**: 2-4GB 기본, 최대 4-8GB
- **스토리지**: 10-20GB (로그, 설정 파일 포함)

#### Docker 배포

**메모리 제약 설정:**
```bash
# Hard limit (필수)
docker run -m 4g upf:latest

# Soft limit (권장)
docker run -m 4g --memory-reservation 2g upf:latest

# CPU 할당
docker run --cpus="1.5" upf:latest
```

**Best Practices:**
1. 배포 전 메모리 요구사항 테스트 수행
2. 호스트 리소스 경합 방지 (placement 최적화)
3. 커널 메모리 제한으로 안정성 확보

### 2.2 엣지 컴퓨팅 환경 (MEC - Multi-access Edge Computing)

#### 배포 위치

```
┌──────────────────────────────────────────────┐
│  5G 네트워크 배포 아키텍처                   │
│                                              │
│  ┌────────────┐      ┌────────────┐         │
│  │  기지국    │──────│ Edge UPF   │         │
│  │  (RAN)     │      │ (엣지 DC)  │         │
│  └────────────┘      └─────┬──────┘         │
│                             │                │
│                      ┌──────▼─────┐         │
│                      │ Central UPF│         │
│                      │ (중앙 DC)  │         │
│                      └────────────┘         │
└──────────────────────────────────────────────┘
```

#### 엣지 환경 특성

| 항목 | Edge DC | Central DC |
|------|---------|------------|
| **위치** | 기지국 근처 | 중앙 데이터센터 |
| **CPU** | 2-8 코어 | 16-64 코어 |
| **메모리** | 4-16GB | 64-256GB |
| **레이턴시** | <5ms | 10-50ms |
| **대역폭** | 제한적 | 높음 |
| **전력** | 제한적 | 충분 |
| **리소스** | 제한적 | 풍부 |

#### 엣지 배포 제약사항

1. **하드웨어 제약**
   - 제한된 CPU 코어 (2-8 코어)
   - 제한된 메모리 (4-16GB)
   - 제한된 스토리지 (100GB-1TB)

2. **네트워크 제약**
   - 백홀 대역폭 제한
   - 모델 업데이트 시 다운타임 최소화 필요

3. **운영 제약**
   - 원격 관리 필요
   - 자동 복구 메커니즘 필수
   - 낮은 유지보수 빈도

---

## 3. ML 모델 배포 런타임 옵션 비교

### 3.1 런타임 옵션 상세 분석

#### Option A: Python + scikit-learn (현재 방식)

**장점:**
- ✅ 개발 속도 빠름
- ✅ 디버깅 쉬움
- ✅ 라이브러리 풍부
- ✅ 유지보수 용이

**단점:**
- ❌ 모델 크기 큼 (Random Forest: 246MB)
- ❌ 메모리 사용량 높음 (1GB+)
- ❌ 추론 속도 느림 (Python 오버헤드)
- ❌ 의존성 많음 (numpy, scipy, pandas 등)

**시스템 요구사항:**
```bash
CPU: x86-64, ARM64
메모리: 최소 1GB (모델 로딩 + 추론)
디스크: 500MB (Python + 패키지)
OS: Linux with Python 3.10+
```

**적합한 환경:**
- 리소스가 풍부한 중앙 DC
- 개발/테스트 환경

#### Option B: ONNX Runtime (권장) ⭐⭐⭐⭐⭐

**장점:**
- ✅ 크로스 플랫폼 (x86-64, ARM64, ARM)
- ✅ 경량 런타임 (10-50MB)
- ✅ 빠른 추론 (C++ API)
- ✅ 최적화된 연산자
- ✅ 양자화 지원 (float32 → float16/int8)
- ✅ 컨테이너 친화적

**단점:**
- ⚠️ 모델 변환 필요 (sklearn → ONNX)
- ⚠️ 디버깅 복잡도 증가

**시스템 요구사항:**
```bash
CPU: x86-64, ARM64, ARM (Cortex-A)
메모리: 최소 512MB
디스크: 50MB (런타임) + 1-5MB (모델)
OS: Linux, Windows, macOS
```

**성능 특성:**
- 추론 시간: <10ms (CPU 기준)
- 메모리: 모델 크기의 2-3배
- 스레드 안전: 멀티 스레드 지원

**적합한 환경:**
- 엣지 DC
- 컨테이너 기반 배포
- **프로덕션 환경 (최적)**

#### Option C: TensorFlow Lite ⭐⭐⭐⭐

**장점:**
- ✅ 매우 경량 (1-5MB 런타임)
- ✅ 모바일/임베디드 최적화
- ✅ 마이크로컨트롤러 지원
- ✅ GPU/NPU 가속 지원
- ✅ 강력한 양자화 (int8, int16)

**단점:**
- ⚠️ scikit-learn 모델 변환 복잡
- ⚠️ TensorFlow/Keras 모델에 최적화
- ⚠️ 트리 기반 모델 지원 제한적

**시스템 요구사항:**
```bash
CPU: x86-64, ARM64, ARM, 마이크로컨트롤러
메모리: 최소 256MB
디스크: 5MB (런타임) + 100KB-1MB (모델)
OS: Linux, Android, iOS, 임베디드 OS
```

**적합한 환경:**
- ARM 기반 엣지 장비
- 리소스가 극도로 제한적인 환경
- 딥러닝 모델 (LSTM) 배포

#### Option D: Native C/C++ ⭐⭐⭐⭐

**장점:**
- ✅ 최소 크기 (50KB-500KB)
- ✅ 최대 성능
- ✅ 제로 의존성
- ✅ 완전한 제어

**단점:**
- ❌ 개발 시간 오래 걸림
- ❌ 유지보수 어려움
- ❌ 모델 업데이트 복잡
- ❌ 플랫폼별 빌드 필요

**적합한 환경:**
- 특수 목적 하드웨어
- 극도의 성능 최적화 필요
- 장기간 안정적 운영

### 3.2 런타임 옵션 비교 표

| 항목 | Python + sklearn | ONNX Runtime | TensorFlow Lite | Native C/C++ |
|------|------------------|--------------|-----------------|--------------|
| **모델 크기** | 246MB | 1-5MB | 100KB-1MB | 50KB-500KB |
| **런타임 크기** | 500MB+ | 10-50MB | 1-5MB | 0MB |
| **메모리 요구** | 1GB+ | 512MB+ | 256MB+ | 128MB+ |
| **추론 시간** | 10-50ms | <10ms | <5ms | <1ms |
| **CPU 지원** | x86-64, ARM64 | x86-64, ARM64, ARM | 모든 플랫폼 | 모든 플랫폼 |
| **개발 난이도** | ⭐ (쉬움) | ⭐⭐⭐ (중간) | ⭐⭐⭐⭐ (어려움) | ⭐⭐⭐⭐⭐ (매우 어려움) |
| **유지보수** | ⭐ (쉬움) | ⭐⭐ (보통) | ⭐⭐⭐ (보통) | ⭐⭐⭐⭐⭐ (어려움) |
| **배포 난이도** | ⭐⭐ (보통) | ⭐⭐ (보통) | ⭐⭐⭐ (중간) | ⭐⭐⭐⭐ (어려움) |
| **권장도** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 4. 시나리오별 배포 전략

### 시나리오 A: 상용 네트워크 장비 (Intel x86 서버)

**환경 특성:**
```
하드웨어: Intel Xeon CPU, 8-16 코어
메모리: 16-32GB RAM
스토리지: 500GB-1TB SSD
OS: Ubuntu 20.04 LTS (Linux 5.4+)
배포: Kubernetes/Docker
```

**추천 배포 방식: ONNX Runtime (컨테이너)**

**모델 최적화 목표:**
- 모델 크기: 5-10MB 이하
- 추론 시간: <10ms
- 메모리 사용: <1GB

**Docker 컨테이너 구성:**
```dockerfile
FROM python:3.10-slim

# ONNX Runtime 설치
RUN pip install onnxruntime==1.16.0

# 모델 복사
COPY model_optimized.onnx /app/model.onnx
COPY inference_server.py /app/

WORKDIR /app
CMD ["python", "inference_server.py"]
```

**리소스 할당:**
```yaml
resources:
  requests:
    cpu: "1000m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "2Gi"
```

### 시나리오 B: 엣지 컴퓨팅 박스 (ARM 기반)

**환경 특성:**
```
하드웨어: ARM Cortex-A72, 2-4 코어
메모리: 4-8GB RAM
스토리지: 64GB-128GB eMMC
OS: Ubuntu Core 20.04 (ARM64)
배포: Docker or Snap
```

**추천 배포 방식: ONNX Runtime 또는 TensorFlow Lite**

**모델 최적화 목표:**
- 모델 크기: 1-5MB 이하 (필수)
- 추론 시간: <50ms
- 메모리 사용: <512MB

**경량화 전략:**
```python
# ONNX 양자화 (float32 → float16)
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='model.onnx',
    model_output='model_quantized.onnx',
    weight_type=QuantType.QUInt8
)
# 크기 감소: 5MB → 1.5MB (70% 감소)
```

### 시나리오 C: 임베디드 네트워크 장비

**환경 특성:**
```
하드웨어: ARM Cortex-A53, 1-2 코어
메모리: 1-2GB RAM
스토리지: 16GB-32GB Flash
OS: 커스텀 임베디드 Linux
배포: 바이너리 실행 파일
```

**추천 배포 방식: TensorFlow Lite Micro 또는 Native C**

**모델 최적화 목표:**
- 모델 크기: <1MB (필수)
- 추론 시간: <100ms
- 메모리 사용: <256MB

**극한 경량화 전략:**
1. 트리 개수 대폭 감소 (500 → 10-50)
2. 트리 깊이 제한 (무제한 → 5-10)
3. int8 양자화 적용
4. Feature selection (12개 → 상위 5개)

---

## 5. 권장 아키텍처

### 5.1 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    5G UPF Container                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Main UPF Process (Go/C)                      │  │
│  │                                                       │  │
│  │  - GTP-U 패킷 처리 (데이터 플레인)                  │  │
│  │  - 세션 관리                                         │  │
│  │  - QoS 제어                                          │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────┐          │  │
│  │  │  Monitoring Agent                      │          │  │
│  │  │  - CPU 메트릭 수집 (1분 간격)          │          │  │
│  │  │  - Feature engineering                 │          │  │
│  │  │  - Historical data buffer (24시간)    │          │  │
│  │  └──────────────┬─────────────────────────┘          │  │
│  └─────────────────┼──────────────────────────────────────┘  │
│                    │                                         │
│                    │ gRPC/REST                               │
│                    ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │      ML Inference Service (Sidecar Container)        │  │
│  │                                                       │  │
│  │  Runtime: ONNX Runtime (C++ API)                     │  │
│  │  Model: model_quantized.onnx (1-5MB)                 │  │
│  │  Memory: ~512MB                                       │  │
│  │  Latency: <10ms                                       │  │
│  │                                                       │  │
│  │  API Endpoints:                                       │  │
│  │  - POST /predict (실시간 예측)                       │  │
│  │  - GET /health (헬스체크)                           │  │
│  │  - GET /metrics (Prometheus)                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Kubernetes 배포 매니페스트

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: upf-with-ml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: upf
  template:
    metadata:
      labels:
        app: upf
    spec:
      containers:
      # Main UPF container
      - name: upf
        image: upf:latest
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        ports:
        - containerPort: 8805
          protocol: UDP
          name: gtpu

      # ML Inference sidecar
      - name: ml-inference
        image: ml-inference-onnx:latest
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: MODEL_PATH
          value: "/models/model_quantized.onnx"
        - name: NUM_THREADS
          value: "2"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true

      volumes:
      - name: models
        configMap:
          name: ml-models
```

### 5.3 API 인터페이스 설계

#### REST API (권장)

```python
# inference_server.py
from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI()

# 모델 로딩 (시작 시 한 번만)
session = ort.InferenceSession("model_quantized.onnx")

@app.post("/predict")
async def predict(features: list[float]):
    """
    CPU load 예측

    Input: 12개 features (lag_1min, lag_5min, ..., is_weekend)
    Output: 다음 1분 평균 CPU load
    """
    input_array = np.array([features], dtype=np.float32)
    outputs = session.run(None, {"input": input_array})
    prediction = float(outputs[0][0])

    return {
        "prediction": prediction,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    # Prometheus 메트릭
    return {
        "inference_count": inference_counter,
        "avg_latency_ms": avg_latency,
        "model_size_mb": model_size
    }
```

#### gRPC API (고성능)

```protobuf
// inference.proto
syntax = "proto3";

service MLInference {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
}

message PredictRequest {
  repeated float features = 1;  // 12개 features
}

message PredictResponse {
  float prediction = 1;          // 예측값
  int64 timestamp = 2;           // Unix timestamp
  float inference_time_ms = 3;   // 추론 시간
}
```

---

## 6. 경량화 실험 결과 (실측)

> 아래 결과는 실제 벤치마크를 통해 측정된 값입니다.
> 스크립트: `scripts/optimize_rf_onnx.py`, `scripts/optimize_rf_pruned_onnx.py`, `scripts/optimize_xgb_onnx.py`

### 6.1 Random Forest ONNX 변환 (원본 모델)

**원본 모델 설정:**
```python
RandomForestRegressor(
    n_estimators=100,      # 트리 100개
    max_depth=None,        # 무제한 깊이
    min_samples_split=2,
    min_samples_leaf=1
)
# 결과: 245.64MB (pkl)
```

**ONNX 변환 + float16 양자화 결과:**

| 모델 | 크기 | MAPE | MAE | RMSE | 추론 시간 (배치) |
|------|------|------|-----|------|-----------------|
| Original (pkl) | 245.64 MB | 0.8913% | 0.4067 | 0.5132 | 39.64ms |
| ONNX (float32) | 137.63 MB | 0.8913% | 0.4067 | 0.5132 | 9.04ms |
| ONNX (float16) | 137.63 MB | 0.8917% | 0.4070 | 0.5135 | 6.88ms |

**핵심 발견:**
- ONNX 변환으로 크기 44% 감소 (245.64MB → 137.63MB)
- float16 양자화는 트리 모델에서 추가 크기 감소 효과 없음 (트리 구조가 크기 지배)
- 정확도 저하는 무시 가능 수준 (+0.0005% MAPE)
- 137MB도 네트워크 장비에는 여전히 큼 → Tree Pruning 필요

### 6.2 Random Forest Tree Pruning + ONNX (실측)

4가지 pruning 설정을 비교하여 크기-정확도 트레이드오프를 실측했습니다:

| 모델 | pkl 크기 | ONNX fp16 크기 | 감소율 | MAPE | Δ MAPE | 추론 레이턴시 |
|------|----------|---------------|--------|------|--------|-------------|
| **Original (n=100, d=∞)** | **245.64 MB** | **137.63 MB** | - | **0.8913%** | - | 23.95ms |
| RF (n=50, d=15) | 26.68 MB | 14.43 MB | 94.1% | 0.8977% | +0.0064% | 0.0081ms |
| RF (n=30, d=12) | 8.41 MB | 4.54 MB | 98.2% | 0.9098% | +0.0185% | 0.0073ms |
| RF (n=20, d=10) | 2.40 MB | 1.29 MB | 99.5% | 0.9441% | +0.0528% | 0.0067ms |
| **RF (n=10, d=8)** | **0.35 MB** | **0.19 MB** | **99.9%** | **1.0112%** | **+0.1199%** | **0.0063ms** |

**Pruning 설정 상세:**
```python
RandomForestRegressor(
    n_estimators=10,       # 트리 10개 (10배 감소)
    max_depth=8,           # 최대 깊이 8
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
# 결과: 0.19MB ONNX fp16, MAPE 1.01%
```

**핵심 발견:**
- Tree Pruning이 트리 모델 경량화의 핵심 (float16보다 훨씬 효과적)
- n=30, d=12가 **Sweet Spot**: 4.54MB에 MAPE 0.91% (원본과 거의 동일)
- n=10, d=8은 **극한 경량화**: 190KB에 MAPE 1.01% (여전히 목표 10% 대비 10배 우수)

### 6.3 XGBoost ONNX 변환 (실측)

**XGBoost는 이미 경량 모델이므로 ONNX 변환만으로 충분했습니다:**

| 모델 | 크기 | MAPE | MAE | RMSE | 추론 레이턴시 (단일 샘플) |
|------|------|------|-----|------|------------------------|
| Original (pkl) | 438.4 KB | 0.9293% | 0.4233 | 0.5329 | 0.8680ms |
| ONNX (float32) | 337.7 KB | 0.9293% | 0.4233 | 0.5329 | 0.0315ms |
| **ONNX (float16)** | **338.0 KB** | **0.9294%** | **0.4233** | **0.5328** | **0.0276ms** |

**핵심 발견:**
- XGBoost는 원본이 이미 438KB로 경량
- ONNX 변환 후 338KB (23% 감소)
- 정확도 변화 사실상 없음 (+0.0001% MAPE)
- 추론 속도 31배 향상 (0.87ms → 0.028ms)
- **RF Original(245.64MB) 대비 99.9% 크기 감소**

### 6.4 전체 모델 비교 (실측 요약)

| 모델 | 크기 | MAPE | 추론 (단일 샘플) | 배포 권장도 |
|------|------|------|-----------------|-----------|
| RF Original (pkl) | 245.64 MB | 0.8913% | 23.95ms | ❌ 너무 큼 |
| RF ONNX fp16 (원본) | 137.63 MB | 0.8917% | 0.028ms | ❌ 여전히 큼 |
| RF Pruned n=30,d=12 fp16 | 4.54 MB | 0.9098% | 0.007ms | ⭐⭐⭐⭐ 높은 정확도 |
| RF Pruned n=10,d=8 fp16 | 0.19 MB | 1.0112% | 0.006ms | ⭐⭐⭐⭐ 최소 크기 |
| **XGBoost ONNX fp16** | **0.33 MB** | **0.9294%** | **0.028ms** | **⭐⭐⭐⭐⭐ 최적 밸런스** |

**권장사항:**
- **일반 배포**: XGBoost ONNX fp16 (338KB, MAPE 0.93%) — 크기와 정확도 최적 밸런스
- **최소 크기 필요**: RF Pruned n=10,d=8 fp16 (190KB, MAPE 1.01%)
- **최고 정확도 필요**: RF Pruned n=30,d=12 fp16 (4.54MB, MAPE 0.91%)

---

## 7. 성능 벤치마크 결과 (실측)

> 벤치마크 환경: Linux 6.17.0, Python 3.12, ONNX Runtime 1.24.4
> 테스트 데이터: 8,640 샘플 (7일), 레이턴시 측정: 단일 샘플 × 1000회 반복

### 7.1 추론 레이턴시 벤치마크 (단일 샘플, 1000회 반복)

| 모델 | Mean (ms) | P50 (ms) | P99 (ms) | 속도 향상 |
|------|-----------|----------|----------|----------|
| RF Original (sklearn) | 23.9487 | 26.2937 | 27.3014 | 1x (기준) |
| RF ONNX fp32 | 0.0301 | 0.0267 | 0.0699 | **796x** |
| RF ONNX fp16 | 0.0277 | 0.0271 | 0.0398 | **865x** |
| RF Pruned n=50,d=15 fp16 | 0.0081 | - | 0.0159 | **2,957x** |
| RF Pruned n=10,d=8 fp16 | 0.0063 | - | 0.0112 | **3,801x** |
| XGBoost Original (sklearn) | 0.8680 | - | 1.8581 | **28x** |
| XGBoost ONNX fp32 | 0.0315 | - | 0.0445 | **760x** |
| XGBoost ONNX fp16 | 0.0276 | - | 0.0460 | **868x** |

**핵심 발견:**
- 모든 ONNX 모델이 <0.05ms 레이턴시로 네트워크 장비 실시간 처리에 충분
- sklearn → ONNX 변환만으로 800배 이상 속도 향상 (Python 인터프리터 오버헤드 제거)
- P99 레이턴시도 0.07ms 이하로 안정적

### 7.2 크기 vs 정확도 트레이드오프 (실측)

| 모델 | 크기 | MAPE (%) | Δ MAPE | 크기 감소율 |
|------|------|----------|--------|-----------|
| RF Original (pkl) | 245.64 MB | 0.8913 | - | - |
| RF ONNX fp16 | 137.63 MB | 0.8917 | +0.0005% | 44.0% |
| RF Pruned n=50,d=15 fp16 | 14.43 MB | 0.8977 | +0.0064% | 94.1% |
| RF Pruned n=30,d=12 fp16 | 4.54 MB | 0.9098 | +0.0185% | 98.2% |
| RF Pruned n=20,d=10 fp16 | 1.29 MB | 0.9441 | +0.0528% | 99.5% |
| RF Pruned n=10,d=8 fp16 | 0.19 MB | 1.0112 | +0.1199% | 99.9% |
| XGBoost (pkl) | 0.43 MB | 0.9293 | +0.0380% | 99.8% |
| **XGBoost ONNX fp16** | **0.33 MB** | **0.9294** | **+0.0381%** | **99.9%** |

### 7.3 벤치마크 재현 스크립트

```bash
# ONNX 변환 (원본 RF)
python scripts/optimize_rf_onnx.py

# Tree Pruning + ONNX (4가지 설정 비교)
python scripts/optimize_rf_pruned_onnx.py

# XGBoost ONNX 변환
python scripts/optimize_xgb_onnx.py
```

결과 파일 위치: `results/optimized/`
- `optimization_report.txt` — RF ONNX 변환 벤치마크
- `rf_pruning_report.txt` — RF Tree Pruning 벤치마크
- `xgb_onnx_report.txt` — XGBoost ONNX 벤치마크

---

## 8. 다음 단계 (Action Items)

### Phase 1: 모델 경량화 ✅ 완료

**완료된 작업:**
```bash
# RF ONNX 변환 (원본 → fp32 → fp16)
python scripts/optimize_rf_onnx.py

# RF Tree Pruning (4가지 설정 비교) + ONNX fp16
python scripts/optimize_rf_pruned_onnx.py

# XGBoost ONNX 변환 + fp16
python scripts/optimize_xgb_onnx.py
```

**생성된 모델 파일 (`results/optimized/`):**
- `rf_model_fp32.onnx` (137.63MB) — RF 원본 ONNX
- `rf_model_fp16.onnx` (137.63MB) — RF 원본 fp16
- `rf_pruned_n50_d15_fp16.onnx` (14.43MB) — RF Pruned
- `rf_pruned_n30_d12_fp16.onnx` (4.54MB) — RF Pruned
- `rf_pruned_n20_d10_fp16.onnx` (1.29MB) — RF Pruned
- `rf_pruned_n10_d8_fp16.onnx` (0.19MB) — RF Pruned (최소)
- `xgb_model_fp16.onnx` (338KB) — **권장 배포 모델**

### Phase 2: 성능 벤치마크 ✅ 완료

벤치마크 리포트 (`results/optimized/`):
- `optimization_report.txt` — RF ONNX 변환 결과
- `rf_pruning_report.txt` — RF Tree Pruning 비교 결과
- `xgb_onnx_report.txt` — XGBoost ONNX 변환 결과

### Phase 3: 배포 준비 (우선순위: 높음, 미완료)

**Task 3.1: ONNX Runtime 추론 서버 구현**
```bash
# FastAPI + ONNX Runtime
python scripts/inference_server_onnx.py \
    --model results/optimized/xgb_model_fp16.onnx \
    --host 0.0.0.0 \
    --port 8080
```

**Task 3.2: Docker 이미지 빌드**
```bash
docker build -t ml-inference-onnx:latest \
    -f deployment/Dockerfile.onnx .
```

**Task 3.3: Kubernetes 매니페스트 작성**
```bash
kubectl apply -f deployment/k8s-deployment.yaml
```

### Phase 4: 실제 환경 테스트 (우선순위: 중간, 미완료)

**Task 4.1: 로컬 UPF 환경 셋업**
```bash
# free5GC 또는 Open5GS 로컬 배포
git clone https://github.com/free5gc/free5gc.git
```

**Task 4.2: ML 서비스 통합 테스트**
```bash
docker-compose -f deployment/docker-compose-upf-ml.yaml up
```

**Task 4.3: 부하 테스트**
```bash
python scripts/load_test.py \
    --target http://localhost:8080/predict \
    --duration 300s \
    --rps 1000
```

---

## 9. 예상 비용 절감 효과

### 9.1 리소스 비용 비교

**현재 (Python + scikit-learn):**
```
CPU: 2 코어
메모리: 4GB
스토리지: 1GB
월 비용: $50-100 (클라우드 VM 기준)
```

**최적화 후 (ONNX Runtime):**
```
CPU: 1 코어
메모리: 1GB
스토리지: 100MB
월 비용: $10-20 (클라우드 VM 기준)
```

**비용 절감:**
- 인프라 비용: 80% 절감
- UPF 당 연간 절감액: $480-960
- 100개 UPF 배포 시: 연간 $48,000-96,000 절감

### 9.2 성능 개선 효과 (실측)

| 항목 | RF Original (pkl) | XGBoost ONNX fp16 | 개선율 |
|------|-------------------|-------------------|--------|
| 모델 크기 | 245.64 MB | 0.33 MB (338KB) | **99.9% ↓** |
| 추론 시간 (단일) | 23.95ms | 0.028ms | **99.9% ↓ (855x)** |
| MAPE | 0.8913% | 0.9294% | +0.04% (무시 가능) |

| 항목 | RF Original (pkl) | RF Pruned n=10 fp16 | 개선율 |
|------|-------------------|---------------------|--------|
| 모델 크기 | 245.64 MB | 0.19 MB (190KB) | **99.9% ↓** |
| 추론 시간 (단일) | 23.95ms | 0.006ms | **99.97% ↓ (3,800x)** |
| MAPE | 0.8913% | 1.0112% | +0.12% (허용 범위) |

**결론:** 성능 저하는 미미(최대 +0.12% MAPE)하지만, 크기 99.9% 감소, 추론 속도 850~3,800배 향상

---

## 10. 리스크 및 제한사항

### 10.1 기술적 리스크

1. **모델 변환 실패** ✅ 해결됨
   - sklearn RF: `skl2onnx`로 변환 성공
   - XGBoost: `onnxmltools`로 변환 성공 (skl2onnx는 XGBoost 미지원, feature name `f%d` 형식 필요)

2. **정확도 저하** ✅ 미미함
   - 실측: 최대 +0.12% MAPE 증가 (RF Pruned n=10,d=8)
   - XGBoost ONNX fp16: +0.0001% MAPE 증가 (사실상 무변화)

3. **런타임 호환성 문제**
   - 리스크: ONNX Runtime이 특정 CPU에서 미지원
   - 완화: 배포 전 타겟 환경에서 테스트 필수

### 10.2 운영 리스크

1. **모델 업데이트**
   - 리스크: 재학습 시 ONNX 변환 파이프라인 재실행 필요
   - 완화: CI/CD 파이프라인 자동화

2. **멀티 플랫폼 지원**
   - 리스크: x86-64, ARM64 각각 빌드 및 테스트 필요
   - 완화: Docker 멀티 아키텍처 이미지 사용

3. **디버깅 복잡도**
   - 리스크: ONNX 모델 디버깅이 Python 모델보다 어려움
   - 완화: 로깅 강화, 테스트 커버리지 확대

### 10.3 제한사항

1. **Feature Engineering 의존성**
   - lag features 계산을 위해 과거 1440분(24시간) 데이터 필요
   - 메모리 버퍼: ~10KB (1440 samples × 8 bytes)

2. **Cold Start 문제**
   - UPF 재시작 시 첫 24시간은 예측 불가
   - 완화: 백업 baseline 모델 사용 또는 중앙 DB에서 히스토리 로드

3. **실시간 재학습 불가**
   - 경량 모델은 추론 전용, 학습은 중앙에서 수행
   - 모델 업데이트는 별도 프로세스 필요

---

## 11. 참고 자료

### 11.1 오픈소스 프로젝트

- **Open5GS**: https://open5gs.org
- **free5GC**: https://free5gc.org
- **ONNX Runtime**: https://onnxruntime.ai
- **TensorFlow Lite**: https://ai.google.dev/edge/litert

### 11.2 기술 문서

- ONNX Format Specification: https://onnx.ai/onnx/
- scikit-learn to ONNX: https://github.com/onnx/sklearn-onnx
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html

### 11.3 관련 논문

- "Deploying Machine Learning Models in 5G Networks" (IEEE)
- "Edge Computing for 5G: A Survey" (ACM Computing Surveys)
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR)

---

## 12. 결론

### 핵심 요약 (실측 기반)

1. **5G UPF는 Linux 컨테이너 기반**으로 배포되며, x86-64 또는 ARM64 환경에서 실행됨
2. **ONNX Runtime 기반 배포**가 가장 적합 — 모든 ONNX 모델에서 <0.05ms 추론 레이턴시 달성
3. **XGBoost ONNX fp16 (338KB, MAPE 0.93%)** 이 최적 배포 모델 — RF Original 대비 99.9% 크기 감소, 정확도 거의 동일
4. **RF Tree Pruning**으로 245.64MB → 0.19MB 경량화 가능 (MAPE 1.01%, 목표 10% 대비 10배 우수)
5. **float16 양자화**는 트리 모델에서 크기 감소 효과 제한적 — 실질적 경량화는 Tree Pruning에서 발생

### 배포 모델 선정 가이드

| 시나리오 | 권장 모델 | 파일 | 크기 | MAPE |
|----------|----------|------|------|------|
| **일반 배포 (추천)** | XGBoost ONNX fp16 | `xgb_model_fp16.onnx` | 338 KB | 0.93% |
| **최소 크기 필요** | RF Pruned n=10,d=8 | `rf_pruned_n10_d8_fp16.onnx` | 190 KB | 1.01% |
| **최고 정확도 필요** | RF Pruned n=30,d=12 | `rf_pruned_n30_d12_fp16.onnx` | 4.54 MB | 0.91% |

### 권장 Action Plan

**단기 — 완료됨 ✅:**
- [x] 런타임 환경 리서치 완료
- [x] Random Forest 경량화 스크립트 작성 (`scripts/optimize_rf_pruned_onnx.py`)
- [x] ONNX 변환 및 float16 양자화 구현 (`scripts/optimize_rf_onnx.py`, `scripts/optimize_xgb_onnx.py`)
- [x] 성능 벤치마크 수행 (레이턴시, 크기, 정확도)

**중기 — 다음 단계:**
- [ ] ONNX Runtime 추론 서버 구현 (FastAPI + ONNX Runtime)
- [ ] Docker 이미지 빌드
- [ ] 로컬 UPF 환경에서 통합 테스트

**장기:**
- [ ] 실제 5G 네트워크 환경에서 파일럿 배포
- [ ] 프로덕션 환경 모니터링 및 최적화
- [ ] Auto-scaling 연동

---

**문서 작성**: 2026-03-19
**최종 업데이트**: 2026-03-19 — 벤치마크 실측 결과 반영
