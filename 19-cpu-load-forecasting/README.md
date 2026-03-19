# UPF CPU Load 예측 시스템

> 머신러닝을 활용한 5G 코어 네트워크 UPF (User Plane Function) CPU 부하 예측 시스템

---

## 프로젝트 개요

5G 코어 네트워크 UPF의 CPU 부하를 1분 단위로 예측하는 시계열 예측 시스템:
- **용량 최적화**: 과도한 프로비저닝 방지
- **장애 예방**: 피크 부하 사전 감지
- **자동 스케일링**: 예측 기반 리소스 자동 할당

**성과**: MAPE 0.89% (목표 10% 대비 11배 우수)

---

## 데이터

### 원본 데이터 (1분 단위 수집)

| 컬럼 | 타입 | 범위 | 설명 |
|------|------|------|------|
| `timestamp` | datetime | - | 수집 시각 |
| `average_cpu_load` | float | 0-100% | 1분 평균 CPU 부하 (**예측 대상**) |
| `peak_cpu_load` | float | 0-100% | 1분 최대 CPU 부하 (평균 대비 1-3% 높음) |
| `active_session_count` | int | 1,000-250,000 | 활성 세션 수 (CPU 부하에 비례) |

### Train / Test 분할

| 구분 | 기간 | 샘플 수 | 일수 |
|------|------|---------|------|
| **Train** | 2024-01-01 00:00 ~ 2024-01-21 23:59 | 30,240 | 21일 |
| **Test** | 2024-01-22 00:00 ~ 2024-01-28 23:59 | 10,080 | 7일 |

- **분할 방식**: 시간순 분할 (Time-based split, 무작위 셔플 없음)
- **데이터 생성**: 합성 데이터 (`data/generate_data.py`) — 일간 패턴: 새벽 4시 최저(20%), 오전 9시 피크(70%), 저녁 7시 최고(80%)

### 예측 과제

**현재 시점(t-1)까지의 데이터를 사용하여 다음 1분(t)의 `average_cpu_load`를 예측**

---

## Feature Engineering

12개의 feature로 구성된 Standard Set을 사용합니다. 구현: [`src/features/engineering.py`](src/features/engineering.py)

| 카테고리 | Features | 설명 |
|----------|----------|------|
| **Lag Features** (6개) | lag_1min, lag_5min, lag_15min, lag_60min, lag_120min, lag_1440min | 과거 시점 CPU 부하 |
| **Rolling Statistics** (3개) | rolling_mean_15min, rolling_std_15min, rolling_mean_60min | 이동 평균/표준편차 |
| **Temporal** (3개) | day_of_week_sin, day_of_week_cos, is_weekend | 순환 시간 인코딩 |

**Feature Importance (Random Forest 기준):**

| 순위 | Feature | 중요도 |
|------|---------|--------|
| 1 | lag_1min | 68.73% |
| 2 | lag_1440min (24시간 전) | 14.60% |
| 3 | rolling_mean_60min | 4.79% |
| 4 | lag_60min | 3.99% |
| 5 | lag_5min | 2.87% |

자세한 내용: [docs/feature-engineering.md](docs/feature-engineering.md)

---

## 모델 성능

### 모델 비교

| 순위 | 모델 | MAPE | Baseline 대비 | 상태 |
|------|------|------|--------------|------|
| 1 | **Random Forest** | **0.8913%** | **+28.44%** | **최고** |
| 2 | **XGBoost** | **0.9293%** | **+25.39%** | 완료 |
| 3 | **LSTM** | **0.9619%** | **+22.77%** | 완료 |
| - | Seasonal Naive (Baseline) | 1.2455% | 기준선 | 완료 |
| - | Prophet | 1.8997% | -52.52% | 실패 |

### 피크 시간대 성능 (19:00-20:00)

| 모델 | 평균 절대 오차 |
|------|---------------|
| LSTM | 0.3273% |
| Random Forest | 0.4866% |
| XGBoost | 0.5386% |

### 전체 모델 비교 (1시간 피크 구간)

![전체 모델 비교](results/all_models_1hour_comparison.png)

*저녁 피크 시간대(18:00-19:00) 1시간 상세 비교. 상단: ML 모델(RF, XGBoost, LSTM), 중간: baseline, 하단: 예측 오차.*

### Random Forest 예측 결과 (48시간 전체)

![RF 48시간 예측](results/random_forest/random_forest_predictions.png)

*48시간(2일) 전체 구간의 실제값(파란색) vs Random Forest 예측값(녹색). 하단 패널은 예측 오차로, 대부분 +/-1% 이내에 분포.*

### Random Forest 예측 결과 (1시간 줌인 - 피크 시간대)

![RF 1시간 상세](results/random_forest/random_forest_1hour_detailed.png)

*저녁 피크 시간대(18:00-19:00) 1분 단위 상세 비교. 실제값(파란색)의 급격한 변동에도 예측값(녹색)이 추세를 잘 추종하며, 오차는 대부분 1% 이내.*

---

## 경량 모델 최적화 (네트워크 장비 배포용)

5G UPF 네트워크 장비에 직접 배포하기 위해 ONNX 변환 및 float16 양자화를 수행했습니다.

### 최적화 결과 요약

| 모델 | 크기 | MAPE | 추론 레이턴시 (단일) | 배포 권장 |
|------|------|------|---------------------|----------|
| RF Original (pkl) | 245.64 MB | 0.8913% | 23.95ms | 너무 큼 |
| RF Pruned n=30,d=12 fp16 | 4.54 MB | 0.9098% | 0.007ms | 높은 정확도 |
| RF Pruned n=10,d=8 fp16 | 0.19 MB | 1.0112% | 0.006ms | 최소 크기 |
| **XGBoost ONNX fp16** | **338 KB** | **0.9294%** | **0.028ms** | **권장** |

- **크기 감소**: 245.64MB -> 338KB (99.9% 감소)
- **속도 향상**: 23.95ms -> 0.028ms (855배)
- **정확도 유지**: MAPE 0.04% 차이 (무시 가능)

자세한 벤치마크: [docs/deployment-runtime-research.md](docs/deployment-runtime-research.md)

---

## 프로덕션 배포

### 권장: XGBoost ONNX (네트워크 장비용)

```python
import onnxruntime as ort
import numpy as np

# 모델 로드 (338KB)
session = ort.InferenceSession('results/optimized/xgb_model_fp16.onnx')

# 실시간 예측 (0.028ms)
features = np.array([[lag_1min, lag_5min, ...]], dtype=np.float32)
prediction = session.run(None, {'float_input': features})[0][0]
```

**요구사항**: CPU 1 코어, RAM 512MB, 레이턴시 <1ms

### 대안: sklearn 기반 (서버 환경용)

```python
import joblib
model = joblib.load('results/random_forest/random_forest_model.pkl')
prediction = model.predict([features])[0]
```

**요구사항**: CPU 2 코어, RAM 4GB, 레이턴시 <25ms

### 고급 전략: 하이브리드 (RF + LSTM)

피크 시간대(18:00-21:00)는 LSTM, 일반 시간대는 RF 사용. 자세한 내용: [docs/final-report.md](docs/final-report.md)

---

## 빠른 시작

### 1. 환경 설정
```bash
cd setup
./run_all_setup.sh  # 원클릭 설치 (UV + 의존성)
source .venv/bin/activate
```

### 2. 데이터 생성
```bash
cd data
python generate_data.py --days 21 --output train.csv
python generate_data.py --days 7 --output test.csv --start-date 2024-01-22
```

### 3. Feature Engineering
```bash
python -m src.features.engineering
```

### 4. 모델 학습
```bash
python scripts/baseline_seasonal_naive.py --plot
python scripts/train_random_forest.py
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/train_prophet.py
```

### 5. 경량 모델 생성
```bash
python scripts/optimize_rf_onnx.py
python scripts/optimize_rf_pruned_onnx.py
python scripts/optimize_xgb_onnx.py
```

### 6. 시각화
```bash
python scripts/visualize_random_forest_detailed.py
python scripts/visualize_xgboost_detailed.py
python scripts/visualize_lstm_detailed.py
python scripts/visualize_all_models_comparison.py
```

---

## 프로젝트 구조

```
19-cpu-load-forecasting/
├── data/                       # 데이터셋
│   ├── train.csv              # 21일 학습 데이터
│   ├── test.csv               # 7일 테스트 데이터
│   ├── processed/             # Feature engineering 완료 데이터
│   ├── generate_data.py       # 합성 데이터 생성기
│   └── split_train_test.py    # Train/test 분할
│
├── src/                       # 소스 코드
│   ├── features/
│   │   └── engineering.py     # Feature engineering (12개 features)
│   └── models/
│       └── __init__.py
│
├── scripts/                   # 학습, 최적화, 시각화 스크립트
│   ├── baseline_seasonal_naive.py
│   ├── train_xgboost.py
│   ├── train_random_forest.py
│   ├── train_lstm.py
│   ├── train_prophet.py
│   ├── optimize_rf_onnx.py         # RF ONNX 변환 + fp16
│   ├── optimize_rf_pruned_onnx.py  # RF Tree Pruning + ONNX
│   ├── optimize_xgb_onnx.py        # XGBoost ONNX 변환 + fp16
│   └── visualize_*_detailed.py
│
├── results/                   # 모델 출력 결과
│   ├── baseline/              # Baseline 결과
│   ├── random_forest/         # Random Forest 결과
│   ├── xgboost/               # XGBoost 결과
│   ├── lstm/                  # LSTM 결과
│   ├── prophet/               # Prophet 결과
│   └── optimized/             # ONNX 경량 모델 + 벤치마크 리포트
│
├── docs/                      # 문서
│   ├── final-report.md                  # 종합 최종 보고서
│   ├── deployment-runtime-research.md   # 5G UPF 배포 환경 리서치 + 벤치마크
│   ├── feature-engineering.md           # Feature engineering 가이드
│   └── baseline-results.md             # Baseline 분석
│
├── setup/                     # 환경 설정
│   ├── run_all_setup.sh       # 원클릭 설치
│   └── pyproject.toml         # UV 의존성
│
├── CLAUDE.md                  # AI용 프로젝트 가이드
└── README.md                  # 이 파일
```

---

## 기술 스택

| 구성 요소 | 기술 | 버전 |
|-----------|------|------|
| 언어 | Python | 3.10+ |
| 패키지 관리 | UV | latest |
| 데이터 처리 | pandas, numpy | 2.0+, 1.24+ |
| ML | scikit-learn, XGBoost | 1.3+, 2.0+ |
| 딥러닝 | PyTorch | 2.0+ |
| 시계열 | Prophet | 1.1+ |
| 모델 경량화 | ONNX Runtime, skl2onnx, onnxmltools | 1.24+, 1.16+, 1.12+ |
| 시각화 | matplotlib, seaborn | 3.7+, 0.12+ |

---

## 문서

| 문서 | 설명 |
|------|------|
| [docs/final-report.md](docs/final-report.md) | 종합 프로젝트 보고서 (모델 비교, 배포 가이드, 비용 분석) |
| [docs/deployment-runtime-research.md](docs/deployment-runtime-research.md) | 5G UPF 런타임 환경 리서치 + ONNX 경량화 벤치마크 |
| [docs/feature-engineering.md](docs/feature-engineering.md) | Feature engineering 상세 가이드 |
| [docs/baseline-results.md](docs/baseline-results.md) | Baseline 모델 분석 |

---

## 향후 로드맵

| Phase | 내용 | 우선순위 | 상태 |
|-------|------|----------|------|
| 1 | 모델 학습 및 평가 | 높음 | 완료 |
| 1.5 | 경량 모델 최적화 (ONNX + fp16) | 높음 | 완료 |
| 2 | 실제 UPF 데이터 검증 | 높음 | 다음 |
| 3 | ONNX 추론 서버 + Docker 배포 | 높음 | 예정 |
| 4 | Feature 확장 (공휴일, 이벤트, Multi-UPF) | 중간 | 예정 |
| 5 | Multi-step 예측 (5/10/30분) | 중간 | 예정 |
| 6 | 실시간 이상 탐지 및 알람 | 높음 | 예정 |

---

## 라이센스

Internal Use Only - Company Confidential

---

**마지막 업데이트**: 2026-03-19
**버전**: 1.2
