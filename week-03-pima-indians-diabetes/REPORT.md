# Pima Indians Diabetes 데이터 인사이트 보고서

## 1. 데이터 개요

| 항목 | 내용 |
|---|---|
| 출처 | Kaggle `kumargh/pimaindiansdiabetescsv` |
| 표본 | 768명 (모두 Pima 인디언 여성, 만 21세 이상) |
| 변수 | 8개 설명변수 + 1개 타깃(Outcome) |
| 타깃 분포 | 당뇨 268명(34.9%), 비당뇨 500명(65.1%) |

**변수 설명**

| 변수 | 의미 |
|---|---|
| Pregnancies | 임신 횟수 |
| Glucose | 경구 포도당 부하 검사(OGTT) 2시간 혈당 |
| BloodPressure | 이완기 혈압 (mmHg) |
| SkinThickness | 삼두근 피부 두께 (mm) |
| Insulin | 2시간 혈청 인슐린 |
| BMI | 체질량지수 |
| DiabetesPedigreeFunction | 당뇨 가족력 점수 |
| Age | 나이 |
| Outcome | 당뇨 진단 여부 (1=당뇨) |

## 2. 데이터 품질 (Data Quality)

- **NaN 결측치: 0개**, **중복 행: 0개** → 형식상 깨끗함.
- 그러나 다음 5개 변수에 `0`이 존재하며, 이는 생리학적으로 불가능하므로 **실질적 결측치**로 판단:

| 변수 | 0값 개수 | 비율 |
|---|---|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

- **이상치(IQR 기준)**: BloodPressure 45개, Insulin 34개, DiabetesPedigreeFunction 29개, BMI 19개 등.
- 이상치 제거 민감도 검증 결과, Glucose 상관계수는 0.495로 **불변**, Insulin은 오히려 0.303 → 0.333으로 **안정적**. 즉 이상치가 상관관계를 인위적으로 만든 것은 아님.

## 3. 상관관계 인사이트

**타깃(Outcome)과의 상관 (Pearson / Spearman)**

| 변수 | Pearson | Spearman | 95% CI (Pearson) |
|---|---|---|---|
| Glucose | 0.495 | 0.483 | [0.44, 0.55] |
| BMI | 0.314 | 0.309 | [0.25, 0.38] |
| Insulin | 0.303 | 0.377 | [0.21, 0.39] |
| SkinThickness | 0.259 | 0.265 | [0.18, 0.34] |
| Age | 0.238 | 0.309 | [0.17, 0.30] |
| Pregnancies | 0.222 | 0.199 | [0.15, 0.29] |
| DiabetesPedigreeFunction | 0.174 | 0.175 | [0.10, 0.24] |
| BloodPressure | 0.171 | 0.177 | [0.10, 0.24] |

**변수 간 상관**

| 변수쌍 | Pearson | 해석 |
|---|---|---|
| SkinThickness ↔ BMI | 0.65 | 체지방 관련 중복 |
| Glucose ↔ Insulin | 0.58 | 대사 관련 |
| Pregnancies ↔ Age | 0.54 | 교란 가능성 |
| BloodPressure ↔ Age | 0.33 | 노화 관련 |

**핵심 해석**

1. **Glucose가 가장 강한 단일 지표**이지만, Outcome 정의 자체가 혈당 기준을 포함하므로 **동어반복(순환)** 성격이 일부 있음. "예측"보다 "진단 일치"에 가까움.
2. **SkinThickness는 BMI와 강하게 얽혀 있음.** BMI 통제 후 순수상관은 0.259 → **0.080 (p=0.06, 유의하지 않음)** 으로 붕괴. 독립적인 예측 변수로 보기 어려움.
3. **Age 효과는 상당 부분 Pregnancies 때문.** Pregnancies 통제 시 0.238 → 0.144.
4. **Insulin은 Spearman이 Pearson보다 높음** → 분포 왜곡(최댓값 846) 영향, 그리고 결측이 49%라 추정 불확실성이 큼.
5. **DiabetesPedigreeFunction, BloodPressure는 약한 신호** (r≈0.17). 통계적으로 유의하나 실질적 기여는 작음.
6. 서로 강하게 상관된 변수들(SkinThickness-BMI, Glucose-Insulin, Pregnancies-Age)은 **다중공선성**을 유발하므로 모델 해석 시 주의.

## 4. 인과 해석 주의문

- 상관은 인과가 아니다. 방향(선행/후행)을 알 수 있는 시간 정보가 없다.
- **역인과/순환**: 당뇨 진단은 Glucose 검사에 기반하므로 Glucose→Outcome은 부분적으로 정의상 관계다.
- **결측 편향**: Insulin·SkinThickness 상관은 각각 전체의 49%·71%에서만 계산됨. 결측이 무작위라는 증거는 없음(Outcome별 결측률 차이는 유의하지 않았으나 이는 "무작위 증명"이 아님).
- **일반화 한계**: Pima 여성 단일 코호트 → 타 인구집단에 그대로 적용 불가.
- **다중비교**: 36개 변수쌍 검정 시 우연한 양성이 섞일 수 있음(단, 핵심 상관은 p가 극히 작아 안전).

## 5. 모델링 전략 (인사이트 → 설계)

| 결정 | 근거 |
|---|---|
| 0값을 NaN으로 변환 후 중앙값 대체 + 결측 지표(indicator) 추가 | 실제 결측이므로 제거 시 49% 손실 |
| 전처리를 Pipeline으로 묶어 train에만 fit | 데이터 누수 방지 |
| 로지스틱 회귀(L2) + 랜덤포레스트 비교 | 해석력 vs 비선형 성능 |
| class_weight='balanced' | 65:35 불균형 완화 |
| 지표: ROC-AUC, Recall, Precision, F1 (Accuracy 단독 사용 금지) | 불균형 데이터에서 정확도는 오해 소지 |
| SkinThickness 단독 신뢰 낮음 → permutation importance로 기여도 확인 | BMI 통제 시 신호 소멸 |

**기대 성능**: 동일 데이터 공개 벤치마크에서 ROC-AUC 약 0.80~0.83 수준이 일반적.

## 6. 다음 단계 제안

1. 다중대체(MICE) 또는 결측 indicator 포함 모델 비교
2. 연령·임신횟수 하위군별 상관 재확인(심슨의 역설 점검)
3. 교차검증·보정 곡선(calibration)으로 확률 신뢰도 평가
4. 외부 데이터로 일반화 검증
