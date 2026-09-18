# intro to data lit

Kaggle 타이타닉 데이터셋으로 배우는 데이터 분석 입문 프로젝트입니다.
데이터 품질 점검부터 시각화, 머신러닝 예측 모델까지 단계별로 다룹니다.

## 데이터

- 출처: Kaggle [`heptapod/titanic`](https://www.kaggle.com/datasets/heptapod/titanic)
- 파일: `train_and_test2.csv` (1,309행 × 28열)
- 다운로드:
  ```bash
  python3 -c "import kagglehub; print(kagglehub.dataset_download('heptapod/titanic'))"
  ```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `eda_quality_check.py` | 결측치·중복·이상치 점검 스크립트 |
| `titanic_quality_check.ipynb` | 데이터 품질 점검 노트북 (입문용) |
| `titanic_model.ipynb` | 시각화 + 생존 예측 모델 노트북 |
| `titanic_correlation_viz.ipynb` | 상관관계 시각화 노트북 (10종 그래프) |
| `titanic_validation.ipynb` | 분석 결론 검증 노트북 (교차검증·신뢰구간·비선형) |
| `titanic_final_model.ipynb` | 인사이트 기반 최종 예측 모델 |
| `INSIGHTS.md` | 인사이트 보고서 (발견·한계·권고) |
| `docs/index.html` | 발표용 HTML 슬라이드 덱 (에디토리얼 테마) |
| `reports/` | 품질 점검 그래프 이미지 |

## 실행 방법

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
python3 -m jupyter notebook
```

브라우저에서 노트북을 열고 `Run → Run All Cells` 를 실행합니다.

## 분석 요약

- **결측치**: `Embarked` 2개(0.15%)뿐, 나머지 없음
- **중복값**: 완전 중복 행 0개
- **이상치**: `Fare` 가 가장 많음 (IQR 171개)
- **불필요 컬럼**: `zero`~`zero.18` 19개는 값이 모두 0이라 제거

## 모델 성능

1차 모델 (테스트 20%):

| 모델 | 정확도 | AUC |
|---|---|---|
| LogisticRegression | 0.752 | 0.759 |
| DecisionTree | 0.760 | 0.769 |
| RandomForest | 0.763 | 0.768 |

→ 정확도는 "전부 사망"(기준선 0.739)과 큰 차이가 없고, 재현율은 0.48로 생존자의 절반을 놓침.

인사이트 반영 최종 모델 (5겹 교차검증):

| 모델 | ROC-AUC | 재현율 | F1 |
|---|---|---|---|
| LogisticRegression | 0.789 | 0.713 | 0.582 |
| **RandomForest (채택)** | **0.805** | **0.725** | **0.605** |
| HistGradientBoosting | 0.802 | 0.453 | 0.526 |

홀드아웃 테스트: 정확도 0.748, AUC 0.782, 재현율 0.662.
→ 피처 엔지니어링(`IsChild`, `FamilySize`)과 클래스 가중치로 재현율을 0.48 → 0.66~0.73 으로 개선.

자세한 내용은 [`INSIGHTS.md`](INSIGHTS.md) 참고.

## 발표 자료

- Live deck: <https://rchoi-v8.github.io/titanic-deck/>
- 소스: <https://github.com/rchoi-v8/titanic-deck>

## 데이터 신뢰성 경고

이 데이터는 **2차 가공본**입니다. `zero.*` 상수 컬럼 19개, Age 인위적 대체 흔적,
`Fare==0` 결측 코딩 등이 있어 원본 타이타닉과 1:1 비교하면 안 됩니다.


## 상관관계 핵심 발견

- `Sex` 가 생존과 가장 강한 상관(+0.40), 여성 생존율이 높음
- `Pclass` 는 음의 상관(-0.25), 등급이 낮을수록 생존율 낮음
- `Fare` 는 Pearson +0.17 / Spearman +0.24 로 치우친 분포의 순위 상관이 더 큼
- **Sex × Pclass 상호작용**: 여성도 3등급은 생존율 0.33, 1·2등급은 0.63~0.66
- **FamilySize 는 비선형**: 4명(0.49)에서 최고, 1명(0.21)·5명 이상은 급감

