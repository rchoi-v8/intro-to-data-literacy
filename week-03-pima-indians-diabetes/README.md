# week-03-pima-indians-diabetes

Kaggle 피마 인디언 당뇨 데이터셋으로 배우는 데이터 분석 입문 프로젝트입니다.
데이터 품질 점검 → 상관관계 시각화 → 인과 주의 → 예측 모델까지 단계별로 다룹니다.

## 데이터

- 출처: Kaggle [`kumargh/pimaindiansdiabetescsv`](https://www.kaggle.com/datasets/kumargh/pimaindiansdiabetescsv)
- 파일: `pima-indians-diabetes.csv` (768행 × 9열, 헤더 없음)
- 다운로드: 스크립트 실행 시 `kagglehub`가 자동 다운로드 (별도 준비 불필요)

## 파일 구성

| 파일 | 설명 |
|---|---|
| `analysis.py` | 결측치·중복·이상치 점검 + 품질 시각화 |
| `correlation.py` | Pearson/Spearman 상관, Pairplot, Clustermap |
| `model.py` | 결측 대체 파이프라인 + 예측 모델 학습·평가 |
| `REPORT.md` | 인사이트 보고서 (발견 · 한계 · 권고) |
| `reports/` | 생성된 그래프 이미지 + 저장 모델(`diabetes_model.joblib`) |

발표 덱 소스는 저장소의 `docs/week-03-pima-indians-diabetes/index.html`.

## 실행 방법

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
python3 analysis.py       # 품질 점검
python3 correlation.py    # 상관관계
python3 model.py          # 예측 모델
```

각 스크립트는 `reports/`에 이미지를 저장합니다.

## 분석 요약

- **결측치**: NaN 0개, 중복 0개. 그러나 `Insulin`(48.7%)·`SkinThickness`(29.6%) 등에 생리학적으로 불가능한 `0`값 다수 → 실질적 결측
- **이상치(IQR)**: BloodPressure 45, Insulin 34, DiabetesPedigreeFunction 29, BMI 19
- **타깃과의 상관**: Glucose .495 > BMI .314 > Insulin .303 > SkinThickness .259 > Age .238
- **변수 간 상관**: SkinThickness↔BMI .65, Glucose↔Insulin .58, Pregnancies↔Age .54
- **핵심 반전**: BMI 통제 후 SkinThickness 순수상관 .259 → **.080**(비유의), Age는 Pregnancies 통제 후 .238 → .144. **Glucose만 .456으로 생존**

## 모델 성능

전처리(0→NaN, 중앙값 대체 + 결측 지표, 표준화)를 `Pipeline`으로 묶어 데이터 누수 차단.
5겹 교차검증: LogisticRegression 0.839 vs RandomForest 0.827 → **로지스틱 회귀 채택**.

홀드아웃 테스트 (20%):

| 모델 | ROC-AUC | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **LogisticRegression (채택)** | **0.818** | 0.727 | 0.685 | 0.597 | 0.638 |

→ 변수 중요도(permutation)에서 Glucose 압도적 1위, SkinThickness는 사실상 0.
상관분석의 경고가 모델에서 재확인됨.

자세한 내용은 [`REPORT.md`](REPORT.md) 참고.

## 발표 자료

- Live deck: <https://rchoi-v8.github.io/intro-to-data-literacy/week-03-pima-indians-diabetes/>
- 소스: `docs/week-03-pima-indians-diabetes/index.html`

## 인과 해석 경고

- 상관은 인과가 아님. 시간 정보가 없어 선후 관계 추론 불가.
- **Glucose → Outcome은 부분적 순환**: 당뇨 진단 자체가 혈당(OGTT) 기준을 포함.
- 표본은 Pima 여성 단일 코호트 → 타 인구집단 일반화 불가.
- Insulin·SkinThickness 상관은 각각 전체의 49%·71%에서만 계산되어 불확실성이 큼.
