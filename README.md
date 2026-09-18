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

## 모델 성능 (테스트 20%)

| 모델 | 정확도 | AUC |
|---|---|---|
| LogisticRegression | 0.752 | 0.759 |
| DecisionTree | 0.760 | 0.769 |
| RandomForest | 0.763 | 0.768 |

생존에 영향을 주는 주요 변수는 `Sex`, `Pclass`, `Fare` 입니다.
