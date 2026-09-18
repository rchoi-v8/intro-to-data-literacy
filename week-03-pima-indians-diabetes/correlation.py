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
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)

# 0을 결측치로 바꾼 데이터 (상관관계 왜곡 방지)
df_clean = df.copy()
df_clean[ZERO_AS_MISSING] = df_clean[ZERO_AS_MISSING].replace(0, np.nan)

features = COLUMNS[:-1]

print("=" * 60)
print("1. 당뇨 여부(Outcome)와의 상관계수 (Pearson)")
print("=" * 60)
pearson_target = df_clean[COLUMNS].corr(numeric_only=True)["Outcome"].drop("Outcome")
print(pearson_target.sort_values(ascending=False).round(3))

print("\n" + "=" * 60)
print("2. 당뇨 여부(Outcome)와의 상관계수 (Spearman)")
print("=" * 60)
spearman_target = df_clean[COLUMNS].corr(method="spearman", numeric_only=True)["Outcome"].drop("Outcome")
print(spearman_target.sort_values(ascending=False).round(3))

print("\n" + "=" * 60)
print("3. 변수 간 상관관계가 높은 쌍 (|r| >= 0.3, Outcome 제외)")
print("=" * 60)
corr_matrix = df_clean[features].corr(numeric_only=True)
pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
pairs.columns = ["변수1", "변수2", "상관계수"]
strong = pairs[pairs["상관계수"].abs() >= 0.3].sort_values("상관계수", key=abs, ascending=False)
print(strong.round(3).to_string(index=False) if len(strong) else "|r| >= 0.3 인 쌍 없음")

# ---------------------------------------------------------------
# 시각화 1: Pearson vs Spearman 상관 히트맵
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(
    df_clean[COLUMNS].corr(numeric_only=True),
    annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
    square=True, ax=axes[0],
)
axes[0].set_title("Pearson 상관계수 (선형 관계)")

sns.heatmap(
    df_clean[COLUMNS].corr(method="spearman", numeric_only=True),
    annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
    square=True, ax=axes[1],
)
axes[1].set_title("Spearman 상관계수 (순위/비선형 관계)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "03_correlation_heatmap.png", dpi=150)
print(f"\n저장 완료: {REPORT_DIR / '03_correlation_heatmap.png'}")

# ---------------------------------------------------------------
# 시각화 2: Outcome과의 상관계수 막대그래프
# ---------------------------------------------------------------
target_corr = pd.DataFrame({
    "Pearson": pearson_target,
    "Spearman": spearman_target,
}).sort_values("Pearson")
target_corr.plot(kind="barh", figsize=(10, 6), color=["steelblue", "darkorange"])
plt.title("당뇨 여부(Outcome)와 각 변수의 상관계수")
plt.xlabel("상관계수")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(REPORT_DIR / "04_target_correlation.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '04_target_correlation.png'}")

# ---------------------------------------------------------------
# 시각화 3: Pairplot (Outcome 색상 구분)
# ---------------------------------------------------------------
pair_cols = ["Glucose", "BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction", "Outcome"]
g = sns.pairplot(
    df_clean[pair_cols].dropna(),
    hue="Outcome",
    palette={0: "steelblue", 1: "crimson"},
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 15},
)
g.fig.suptitle("주요 변수 산점도 행렬 (Outcome 색상)", y=1.02)
g.savefig(REPORT_DIR / "05_pairplot.png", dpi=130)
print(f"저장 완료: {REPORT_DIR / '05_pairplot.png'}")

# ---------------------------------------------------------------
# 시각화 4: 핵심 변수 vs Glucose 산점도 + 회귀선
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.regplot(data=df_clean, x="Glucose", y="BMI", ax=axes[0],
            scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "crimson"})
axes[0].set_title("Glucose vs BMI")
sns.regplot(data=df_clean, x="Age", y="Glucose", ax=axes[1],
            scatter_kws={"alpha": 0.4, "s": 15}, line_kws={"color": "crimson"})
axes[1].set_title("Age vs Glucose")
plt.tight_layout()
plt.savefig(REPORT_DIR / "06_scatter_regression.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '06_scatter_regression.png'}")

# ---------------------------------------------------------------
# 시각화 5: Clustermap (비슷한 변수끼리 묶어서 보기)
# ---------------------------------------------------------------
cluster = sns.clustermap(
    df_clean[COLUMNS].corr(numeric_only=True).fillna(0),
    annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
    figsize=(10, 9),
)
cluster.fig.suptitle("변수 클러스터맵 (상관관계 기반 그룹화)", y=1.02)
cluster.savefig(REPORT_DIR / "07_clustermap.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '07_clustermap.png'}")

plt.show()
