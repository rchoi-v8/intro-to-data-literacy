from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "AppleGothic", "Apple SD Gothic Neo", "NanumGothic",
    "Malgun Gothic", "Noto Sans CJK KR", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = Path(
    kagglehub.dataset_download("kumargh/pimaindiansdiabetescsv")
) / "pima-indians-diabetes.csv"
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

print("=" * 60)
print("1. 데이터 기본 정보")
print("=" * 60)
print(f"행 개수: {df.shape[0]}, 열 개수: {df.shape[1]}\n")
print(df.head())
print("\n기초 통계량")
print(df.describe().T)

print("\n" + "=" * 60)
print("2. 결측치(NaN) 검증")
print("=" * 60)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "NaN 결측치 없음")

zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = (df[zero_cols] == 0).sum()
print("\n0으로 기록된 값 (의학적으로 불가능 -> 결측치로 의심)")
print(zero_counts)

print("\n" + "=" * 60)
print("3. 중복값 검증")
print("=" * 60)
dup_count = df.duplicated().sum()
print(f"중복 행 개수: {dup_count}")
if dup_count > 0:
    print(df[df.duplicated(keep=False)].sort_values(COLUMNS[:3]).head(10))

print("\n" + "=" * 60)
print("4. 이상치(IQR) 검증")
print("=" * 60)
outlier_summary = {}
for col in COLUMNS[:-1]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = count
    print(f"{col:<26} 하한 {lower:8.2f} / 상한 {upper:8.2f} -> 이상치 {count}개")

print("\n" + "=" * 60)
print("5. 시각화 저장")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(
    df[COLUMNS[:-1]].isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis",
    ax=axes[0, 0],
)
axes[0, 0].set_title("결측치(NaN) 분포")

zero_ratio = (df[zero_cols] == 0).mean().sort_values()
sns.barplot(x=zero_ratio.values, y=zero_ratio.index, ax=axes[0, 1], hue=zero_ratio.index, palette="Reds_r", legend=False)
axes[0, 1].set_title("0값 비율 (결측치 의심)")
axes[0, 1].set_xlabel("비율")

melted = df[COLUMNS[:-1]].melt(var_name="변수", value_name="값")
sns.boxplot(data=melted, x="변수", y="값", ax=axes[1, 0], hue="변수", palette="Set2", legend=False)
axes[1, 0].set_title("변수별 이상치 (Boxplot)")
axes[1, 0].tick_params(axis="x", rotation=45)

sns.countplot(x="Outcome", data=df, ax=axes[1, 1], hue="Outcome", palette="Set1", legend=False)
axes[1, 1].set_title("당뇨 여부(Outcome) 분포")

plt.tight_layout()
plt.savefig(REPORT_DIR / "01_data_quality.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '01_data_quality.png'}")

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes2.flatten(), zero_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"{col} 분포")
plt.tight_layout()
plt.savefig(REPORT_DIR / "02_distributions.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '02_distributions.png'}")

plt.show()
