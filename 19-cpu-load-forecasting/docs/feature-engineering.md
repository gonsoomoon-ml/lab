# Feature Engineering Guide

CPU Load Forecasting 프로젝트의 feature engineering 가이드입니다.

## 목차

1. [개요](#개요)
2. [Feature Set 구성](#feature-set-구성)
3. [Feature 설명](#feature-설명)
4. [구현 방법](#구현-방법)
5. [데이터 전처리 결과](#데이터-전처리-결과)
6. [사용 방법](#사용-방법)

---

## 개요

### 설계 원칙

- **단변량 시계열 예측**: `average_cpu_load` 값만 사용
- **시간 정보 최소화**: `day_of_week`만 사용 (hour, minute 제외)
- **Standard Set**: 12개 features (성능과 복잡도의 균형)

### 제외된 정보

- `peak_cpu_load`: 다른 컬럼 사용 안 함
- `active_session_count`: 다른 컬럼 사용 안 함
- `hour`, `minute`: 시간 정보 제외 (lag features로 대체)

---

## Feature Set 구성

### Standard Set (총 12개 features)

| Category | Features | Count |
|----------|----------|-------|
| **Lag Features** | 1, 5, 15, 60, 120, 1440분 전 값 | 6 |
| **Rolling Statistics** | 15분/60분 평균, 15분 표준편차 | 3 |
| **Temporal** | day_of_week (sin/cos), is_weekend | 3 |
| **Total** | | **12** |

### 다른 Feature Set 옵션

**Minimal Set (6 features)** - 빠른 프로토타이핑용
```
lag_1min, lag_60min, lag_1440min
rolling_mean_15min
day_of_week_sin, day_of_week_cos
```

**Full Set (28 features)** - 최고 성능 추구
```
Standard Set + 추가 lags, EMA, interactions 등
```

---

## Feature 설명

### 1. Lag Features (과거 값)

과거 특정 시점의 CPU load 값을 feature로 사용합니다.

| Feature | Lag | 의미 |
|---------|-----|------|
| `average_cpu_load_lag_1min` | 1분 | 바로 직전 값 (단기 추세) |
| `average_cpu_load_lag_5min` | 5분 | 최근 변화 감지 |
| `average_cpu_load_lag_15min` | 15분 | 단기 패턴 |
| `average_cpu_load_lag_60min` | 60분 | 1시간 전 값 (시간대 패턴) |
| `average_cpu_load_lag_120min` | 120분 | 2시간 전 값 |
| `average_cpu_load_lag_1440min` | 1440분 | **1일 전 같은 시간 (계절성)** |

**중요:** `lag_1440min`은 seasonal naive baseline의 핵심 feature입니다.

### 2. Rolling Statistics (이동 통계량)

최근 N분간의 통계값을 feature로 사용합니다.

| Feature | Window | 의미 |
|---------|--------|------|
| `average_cpu_load_rolling_mean_15min` | 15분 | 최근 15분 평균 (단기 추세) |
| `average_cpu_load_rolling_std_15min` | 15분 | 최근 15분 변동성 |
| `average_cpu_load_rolling_mean_60min` | 60분 | 최근 1시간 평균 (중기 추세) |

**역할:**
- Rolling mean: 노이즈 제거, 추세 파악
- Rolling std: 변동성 측정 (안정적 vs 불안정)

### 3. Temporal Features (시간 정보)

요일 정보만 사용합니다.

| Feature | 범위 | 의미 |
|---------|------|------|
| `day_of_week_sin` | [-1, 1] | 요일의 cyclical encoding (sin) |
| `day_of_week_cos` | [-1, 1] | 요일의 cyclical encoding (cos) |
| `is_weekend` | {0, 1} | 주말 여부 (토/일=1, 평일=0) |

**Cyclical Encoding 이유:**

요일은 순환하는 값입니다 (일요일 다음은 월요일). 숫자로 인코딩하면 (0=월, 6=일) 모델이 "6과 0이 실제로는 가깝다"는 것을 학습하기 어렵습니다.

```python
# 나쁜 예: 단순 숫자 인코딩
day_of_week = 6  # 일요일
# 모델이 보기엔: 월요일(0)과 일요일(6)이 멀리 떨어져 있음

# 좋은 예: Cyclical encoding
day_of_week_sin = sin(2π × 6 / 7)
day_of_week_cos = cos(2π × 6 / 7)
# 모델이 보기엔: 일요일과 월요일이 가까움
```

---

## 구현 방법

### 코드 위치

```
src/features/engineering.py
```

### 핵심 함수

#### 1. `engineer_features(df)`

전체 feature engineering 파이프라인

```python
from src.features.engineering import engineer_features

# 입력: 원본 데이터 (timestamp, average_cpu_load)
df_engineered = engineer_features(df)

# 출력: 12개 features 추가된 데이터
```

#### 2. `prepare_train_test(train_df, test_df)`

Train/Test 데이터 준비 (NaN 제거 포함)

```python
from src.features.engineering import prepare_train_test

X_train, y_train, X_test, y_test, feature_names = prepare_train_test(
    train_df, test_df
)
```

#### 3. `get_feature_columns()`

Feature 컬럼명 리스트 반환

```python
from src.features.engineering import get_feature_columns

features = get_feature_columns()
# ['average_cpu_load_lag_1min', 'average_cpu_load_lag_5min', ...]
```

---

## 데이터 전처리 결과

### 샘플 수 변화

| Dataset | 원본 | Feature 생성 후 | 제거된 샘플 | 비율 |
|---------|------|----------------|------------|------|
| **Train** | 30,240 | 28,800 | 1,440 | 95.2% |
| **Test** | 10,080 | 8,640 | 1,440 | 85.7% |

**제거 이유:**
- `lag_1440min` (1일 전 값) feature를 생성하려면 1일치 데이터가 필요
- 따라서 첫 1일(1,440분)은 NaN이 되어 제거됨

### 데이터 크기

```
Train: 28,800 samples × 12 features
Test:  8,640 samples × 12 features
```

### 주말/평일 분포

**Train:**
- 평일 (is_weekend=0): 20,160 samples (70%)
- 주말 (is_weekend=1): 8,640 samples (30%)

**Test:**
- 평일 (is_weekend=0): 5,760 samples (67%)
- 주말 (is_weekend=1): 2,880 samples (33%)

### 통계

**Target (average_cpu_load):**
```
평균:      54.57%
표준편차:  18.72%
최소:      18.52%
최대:      81.70%
```

**Lag Features:** Target과 동일한 분포 (시간만 다름)

**Rolling Statistics:**
- Rolling mean: Target과 유사 (약간 부드러움)
- Rolling std: 평균 ~0.5-1% (변동성)

**Temporal Features:**
- day_of_week_sin/cos: [-1, 1] 균등 분포
- is_weekend: 0이 70%, 1이 30%

---

## 사용 방법

### 1. Feature Engineering 실행

```bash
# 방법 1: 스크립트 직접 실행
cd /home/ubuntu/lab/19-cpu-load-forecasting
source .venv/bin/activate
python src/features/engineering.py \
    --train-data data/train.csv \
    --test-data data/test.csv \
    --output-dir data/processed

# 결과: data/processed/train_processed.csv, test_processed.csv 생성
```

```python
# 방법 2: 모듈로 import
from src.features.engineering import prepare_train_test
import pandas as pd

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

X_train, y_train, X_test, y_test, features = prepare_train_test(
    train_df, test_df
)
```

### 2. 생성된 데이터 사용

```python
import pandas as pd

# 전처리된 데이터 로드
train = pd.read_csv('data/processed/train_processed.csv')
test = pd.read_csv('data/processed/test_processed.csv')

# Feature와 Target 분리
X_train = train.drop(columns=['target'])
y_train = train['target']

X_test = test.drop(columns=['target'])
y_test = test['target']

# 모델 학습
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
```

### 3. Feature 중요도 확인

```python
import matplotlib.pyplot as plt

# 모델 학습 후
importances = model.feature_importances_
features = X_train.columns

# 시각화
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Ranking')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

---

## Feature 선택 가이드

### 어떤 Feature Set을 선택할까?

**Minimal Set (6 features) 사용 시:**
- ✅ 빠른 프로토타이핑
- ✅ 학습 속도 빠름
- ❌ 성능 제한적

**Standard Set (12 features) 사용 시:** ⭐ **추천**
- ✅ 성능과 복잡도의 균형
- ✅ 대부분의 ML 모델에 적합
- ✅ 해석 가능성 좋음

**Full Set (28 features) 사용 시:**
- ✅ 최고 성능 가능
- ❌ 과적합 위험
- ❌ 학습 시간 증가
- 💡 XGBoost, Deep Learning에 적합

---

## 주의사항

### 1. Data Leakage 방지

**❌ 나쁜 예:**
```python
# 전체 데이터에 feature engineering 후 split
df_all = engineer_features(df_all)
train, test = train_test_split(df_all)  # WRONG!
```

**✅ 좋은 예:**
```python
# Split 후 각각 feature engineering
train, test = train_test_split(df)
train_fe = engineer_features(train)
test_fe = engineer_features(test)
```

### 2. Rolling Statistics 주의

Rolling statistics는 현재 시점까지의 데이터만 사용해야 합니다 (미래 데이터 사용 금지).

현재 구현은 안전합니다:
```python
# 올바른 구현 (현재 구현)
df['rolling_mean_15min'] = df['average_cpu_load'].rolling(15).mean()
# → 현재 시점 포함 이전 15분 평균
```

### 3. NaN 처리

Lag features로 인해 초반 데이터는 NaN이 됩니다.

현재 구현:
- `dropna()`로 NaN 제거
- Train에서 1,440개 (1일) 제거
- Test에서 1,440개 (1일) 제거

대안:
- Forward fill: `fillna(method='ffill')` (비추천 - 미래 정보 유출)
- Backward fill: `fillna(method='bfill')` (비추천)
- Mean imputation: 초기값 평균으로 채움 (가능)

---

## 성능 예상

### Baseline과 비교

**Seasonal Naive (lag_1440min만 사용):**
- MAPE: 1.25%
- 매우 강력한 baseline

**Standard Set (12 features):**
- 예상 MAPE: 0.8-1.2%
- Seasonal naive보다 약간 개선 (20-40%)

### Feature별 예상 기여도

| Feature | 중요도 예상 | 이유 |
|---------|-----------|------|
| `lag_1440min` | ⭐⭐⭐⭐⭐ | 일별 패턴 (가장 중요) |
| `lag_1min` | ⭐⭐⭐⭐ | 최근 값 (추세) |
| `rolling_mean_60min` | ⭐⭐⭐ | 시간대 평균 |
| `day_of_week_sin/cos` | ⭐⭐⭐ | 요일 패턴 |
| `rolling_std_15min` | ⭐⭐ | 변동성 정보 |
| `lag_5min, lag_15min` | ⭐⭐ | 단기 변화 |
| `lag_60min, lag_120min` | ⭐ | 보조적 정보 |

---

## 다음 단계

Feature engineering 완료 후:

1. **모델 학습**
   - XGBoost (추천)
   - Random Forest
   - Linear Regression

2. **Feature Importance 분석**
   - 어떤 feature가 중요한지 확인
   - 불필요한 feature 제거

3. **하이퍼파라미터 튜닝**
   - Grid search
   - Random search

4. **모델 평가**
   - MAPE < 1.25% (baseline 이하) 목표
   - Error 분석 (시간대별, 요일별)

---

## 참고 자료

### 코드 파일

- `src/features/engineering.py`: Feature engineering 구현
- `data/processed/`: 전처리된 데이터 저장 위치

### 관련 문서

- `docs/data-summary-ko.md`: 데이터셋 분석
- `docs/research-ko.md`: 연구 방법론
- `CLAUDE.md`: 프로젝트 가이드

### 외부 참고

- [Scikit-learn Feature Engineering](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Time Series Feature Engineering](https://www.kaggle.com/code/ryanholbrook/time-series-as-features)
- [Lag Features in Time Series](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

---

**문서 버전:** 1.0
**작성일:** 2024-03-18
**마지막 업데이트:** 2024-03-18
