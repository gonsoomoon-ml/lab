# Baseline Model Results

CPU Load Forecasting 프로젝트의 baseline 모델 성능 문서입니다. 향후 개발할 ML 모델들은 이 baseline을 기준으로 평가됩니다.

## 목차

1. [Baseline 모델 개요](#baseline-모델-개요)
2. [성능 지표](#성능-지표)
3. [시각화 분석](#시각화-분석)
4. [에러 분석](#에러-분석)
5. [인사이트](#인사이트)
6. [모델 비교](#모델-비교)

---

## Baseline 모델 개요

### Seasonal Naive Forecasting

**모델 이름:** Seasonal Naive (계절성 단순 예측)

**예측 방법:**
```python
prediction(t) = actual(t - 1440)
```
- 현재 시점(t)의 CPU load를 예측할 때
- 정확히 1일 전(1440분 전) 같은 시간의 실제 값을 사용

**선택 이유:**
- CPU load는 강한 일별 패턴을 보임 (4AM 최저, 7PM 최고)
- 요일별 패턴도 존재 (주말 vs 평일)
- 과거 같은 시간대의 값이 좋은 예측 기준이 됨

**구현 위치:**
```
scripts/baseline_seasonal_naive.py
```

---

## 성능 지표

### 테스트 데이터

**기간:** 2024-01-22 00:00:00 ~ 2024-01-28 23:59:00 (7일)
**샘플 수:** 10,080개 (1분 단위)
**평가 지표:** MAE, RMSE, MAPE

### 결과

| Metric | Value | 평가 |
|--------|-------|------|
| **MAPE** | **1.2455%** | ✓✓ Excellent (목표 10% 대비) |
| **MAE** | **0.5678%** | ✓✓ Very Good |
| **RMSE** | **0.7107%** | ✓✓ Very Good |

### 목표 대비 성과

**프로젝트 목표:** MAPE < 10%

**Baseline 성능:**
- MAPE 1.2455% → **목표의 12.5% 수준**
- 목표를 **8배 이상 초과 달성**

**의미:**
- Baseline이 이미 매우 우수한 성능
- ML 모델은 1.25% 미만을 목표로 해야 함
- 개선 여지가 작음 (어려운 도전)

---

## 시각화 분석

### 1. 전체 패턴 (48시간)

**파일:** `results/baseline_seasonal_naive_plot.png`

**관찰 내용:**
- 실제값(파란선)과 예측값(주황선)이 거의 겹침
- 일별 패턴이 매우 잘 반복됨
- 오차 분포가 정규분포 형태 (중앙 집중)

### 2. 1시간 상세 뷰

**파일:** `results/baseline_1hour_detailed.png`

**기간:** 19:00-20:00 (저녁 피크 타임)

**통계:**
- 실제 CPU 범위: 77.48% - 80.75%
- 예측 CPU 범위: 77.33% - 81.12%
- 평균 절대 오차: 0.5568%
- 최대 오차: 2.0600%

**관찰 내용:**
- 분 단위 변동(노이즈)이 대부분의 오차 원인
- 예측선이 실제값을 잘 추적함
- 급격한 변화 구간에서 약간의 lag 발생

### 3. 30분 초상세 뷰

**파일:** `results/baseline_30min_ultra_detailed.png`

**기간:** 19:00-19:30

**관찰 내용:**
- 실제값이 예측값보다 더 많은 "wiggle" (변동)
- 이는 데이터 생성 시 추가된 랜덤 노이즈
- 오차의 대부분은 예측 불가능한 노이즈

---

## 에러 분석

### 에러 분포

**평균 절대 오차:** 0.50%
**표준편차:** 0.63%
**최대 양의 오차 (under-prediction):** +2.06%
**최대 음의 오차 (over-prediction):** -1.71%

**95% 신뢰구간:** ±1.26%

### 에러 특성

**1. 정규 분포**
- 에러가 0을 중심으로 정규분포
- 체계적인 bias 없음 (unbiased prediction)
- 양의 오차와 음의 오차가 균형

**2. 작은 크기**
- 대부분의 오차가 ±1% 이내
- CPU load 범위(18-82%)에 비해 매우 작음
- 상대적 오차율 1.25%

**3. 랜덤 노이즈 기반**
- 합성 데이터의 랜덤 노이즈(σ=0.5%)가 주요 원인
- 어제와 오늘의 노이즈 차이 = 오차
- 패턴 자체는 완벽하게 매칭됨

### 시간대별 에러 (예상)

| 시간대 | 예상 에러 | 이유 |
|--------|-----------|------|
| **새벽 (3-5AM)** | 낮음 | CPU load 안정적, 변동 적음 |
| **오전 (7-9AM)** | 중간 | 급격한 상승 구간 |
| **점심 (12-2PM)** | 낮음 | 안정적 구간 |
| **저녁 (6-8PM)** | 중간 | 피크 구간, 변동 있음 |
| **심야 (10PM-12AM)** | 낮음 | 하락 후 안정화 |

---

## 인사이트

### 1. 일별 패턴의 강도

Seasonal naive가 1.25% MAPE를 달성한 것은 **일별 패턴이 매우 강하다**는 증거입니다.

**의미:**
- "어제 같은 시간의 값"만으로 98.75% 정확도
- CPU load가 매우 규칙적으로 반복됨
- 외부 변수(요일, 계절 등)의 영향이 작음 (합성 데이터의 특성)

### 2. ML 모델의 과제

**도전 과제:**
- Baseline이 이미 매우 강력함
- 개선 여지가 1.25% → 1.0% 수준 (20% 개선)
- 노이즈 예측이 핵심

**성공 조건:**
- 단기 추세 학습 (최근 몇 분의 변화)
- 요일 패턴 활용 (주말 vs 평일)
- Rolling statistics로 변동성 감지

### 3. Feature Engineering의 중요성

Baseline이 사용하는 것:
- `lag_1440min` (1일 전 값) 하나만

ML 모델이 추가로 사용할 것:
- Short-term lags: `lag_1min`, `lag_5min` (단기 추세)
- Rolling stats: `rolling_mean_15min`, `rolling_std_15min` (변동성)
- Temporal: `day_of_week` (요일 패턴)

**예상 기여도:**
- `lag_1440min`: 80-90% (기본 패턴)
- 나머지 features: 10-20% (미세 조정)

### 4. 실제 데이터에서의 차이

**합성 데이터:**
- 완벽한 일별 반복
- 노이즈만 랜덤
- Baseline MAPE 1.25%

**실제 UPF 데이터 (예상):**
- 날씨, 이벤트, 장애 등의 영향
- 주중/주말 차이 더 큼
- 트렌드 변화 (점진적 증가/감소)
- Baseline MAPE 3-5% 예상

---

## 모델 비교

### 성능 비교 테이블

| Model | MAPE (%) | MAE | RMSE | Improvement | Status |
|-------|----------|-----|------|-------------|--------|
| **Seasonal Naive** | **1.2455** | **0.5678** | **0.7107** | Baseline | ✓ Complete |
| Linear Regression | - | - | - | - | 🔜 Planned |
| Random Forest | - | - | - | - | 🔜 Planned |
| XGBoost | - | - | - | - | 🔜 Planned |
| LightGBM | - | - | - | - | 📋 Optional |
| LSTM | - | - | - | - | 📋 Optional |
| Prophet | - | - | - | - | 📋 Optional |

**Improvement 계산 방법:**
```
Improvement = (Baseline_MAPE - Model_MAPE) / Baseline_MAPE × 100%

예시: Model MAPE = 1.0%
Improvement = (1.2455 - 1.0) / 1.2455 × 100% = 19.7%
```

### 모델 개발 순서

1. **Linear Regression** - 가장 단순한 ML 모델
2. **Random Forest** - Feature importance 분석
3. **XGBoost** - 최고 성능 기대
4. **LightGBM** (선택) - XGBoost 대안
5. **LSTM** (선택) - 시퀀스 학습
6. **Prophet** (선택) - 시계열 특화

### 성공 기준

| 등급 | MAPE 범위 | 평가 | Improvement |
|------|-----------|------|-------------|
| **S** | < 1.0% | 탁월 | > 20% |
| **A** | 1.0-1.1% | 매우 좋음 | 12-20% |
| **B** | 1.1-1.2% | 좋음 | 4-12% |
| **C** | 1.2-1.3% | 보통 | 0-4% |
| **F** | > 1.3% | 실패 | Baseline보다 나쁨 |

---

## 실행 방법

### Baseline 재실행

```bash
cd /home/ubuntu/lab/19-cpu-load-forecasting
source .venv/bin/activate

python scripts/baseline_seasonal_naive.py \
    --train-data data/train.csv \
    --test-data data/test.csv \
    --plot
```

### 결과 파일

```
results/
├── baseline_seasonal_naive_predictions.csv  # 전체 예측값
├── baseline_seasonal_naive_metrics.txt      # 성능 지표
├── baseline_seasonal_naive_plot.png         # 48시간 시각화
├── baseline_1hour_detailed.png              # 1시간 상세
└── baseline_30min_ultra_detailed.png        # 30분 초상세
```

---

## 다음 단계

### 1. ML 모델 개발

**우선순위 1: XGBoost**
- Tree-based 모델로 feature importance 확인
- 빠른 학습 속도
- 예상 MAPE: 0.9-1.1%

**우선순위 2: Random Forest**
- 비교 대상으로 좋음
- Feature importance 분석
- 예상 MAPE: 1.0-1.2%

**우선순위 3: Linear Regression**
- 가장 단순한 baseline
- Feature 선형 관계 확인
- 예상 MAPE: 1.1-1.3%

### 2. Feature Importance 분석

ML 모델 학습 후:
- 어떤 feature가 가장 중요한가?
- `lag_1440min`이 압도적일 것으로 예상
- 나머지 features의 기여도는?

### 3. 에러 분석

ML 모델의 에러가 baseline과 어떻게 다른가?
- 어떤 시간대에서 더 좋은가?
- 어떤 요일에서 더 좋은가?
- 변동성이 큰 구간에서의 성능은?

### 4. 앙상블

여러 모델을 결합:
```python
ensemble = 0.5 * xgboost_pred + 0.3 * rf_pred + 0.2 * seasonal_naive
```

---

## 참고 정보

### 관련 파일

- 코드: `scripts/baseline_seasonal_naive.py`
- 결과: `results/baseline_*`
- Feature engineering: `docs/feature-engineering.md`

### 성능 지표 정의

**MAE (Mean Absolute Error):**
```
MAE = (1/n) × Σ|actual - predicted|
단위: % (CPU load의 백분율 포인트)
```

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt((1/n) × Σ(actual - predicted)²)
단위: % (MAE보다 큰 오차에 페널티)
```

**MAPE (Mean Absolute Percentage Error):**
```
MAPE = (1/n) × Σ|actual - predicted| / actual × 100%
단위: % (상대적 오차율, 주요 지표)
```

### 왜 MAPE를 주요 지표로 사용하는가?

1. **Scale-independent**: CPU load 범위(18-82%)에 관계없이 비교 가능
2. **해석 쉬움**: "1.25% 오차"가 직관적
3. **업계 표준**: 시계열 예측에서 널리 사용
4. **비즈니스 친화적**: 경영진에게 설명하기 쉬움

단점: actual이 0에 가까울 때 불안정 (본 프로젝트에서는 해당 없음)

---

**문서 버전:** 1.0
**작성일:** 2024-03-18
**마지막 업데이트:** 2024-03-18
**다음 업데이트 예정:** ML 모델 결과 추가 시
