import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = "/Users/remchoi/.cache/kagglehub/datasets/heptapod/titanic/versions/1/train_and_test2.csv"
REPORT_DIR = "reports"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="AppleGothic")


def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"2urvived": "Survived"})
    return df


def check_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    ratio = (missing / len(df) * 100).round(2)

    print("\n=== 1. 결측치 (Missing Values) ===")
    if missing.empty:
        print("결측치가 없습니다.")
        return
    summary = pd.DataFrame({"결측치 개수": missing, "비율(%)": ratio})
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    summary["결측치 개수"].plot(kind="bar", ax=axes[0], color="salmon")
    axes[0].set_title("컬럼별 결측치 개수")
    axes[0].set_ylabel("개수")

    sns.heatmap(df[missing.index].isnull(), cbar=False, cmap="Reds", ax=axes[1])
    axes[1].set_title("결측치 위치")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "01_missing.png"), dpi=120)
    plt.close()


def check_duplicates(df):
    print("\n=== 2. 중복값 (Duplicates) ===")
    dup_count = df.duplicated().sum()
    print(f"완전 중복 행 개수: {dup_count}")

    id_dup = df["Passengerid"].duplicated().sum()
    print(f"Passengerid 중복 개수: {id_dup}")

    if dup_count > 0:
        print(df[df.duplicated(keep=False)].sort_values("Passengerid"))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["중복 행", "고유 행"], [dup_count, len(df) - dup_count],
           color=["tomato", "steelblue"])
    ax.set_title("중복 vs 고유 행")
    ax.set_ylabel("행 개수")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "02_duplicates.png"), dpi=120)
    plt.close()


def check_outliers_iqr(df, columns):
    print("\n=== 3. 이상치 (Outliers, IQR 방식) ===")
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append([col, q1, q3, iqr, lower, upper, count, round(count / len(df) * 100, 2)])

    result = pd.DataFrame(
        rows,
        columns=["컬럼", "Q1", "Q3", "IQR", "하한", "상한", "이상치 수", "비율(%)"],
    )
    print(result.to_string(index=False))

    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 5))
    for ax, col in zip(axes, columns):
        sns.boxplot(y=df[col], ax=ax, color="lightseagreen")
        ax.set_title(f"{col} 박스플롯")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "03_outliers_boxplot.png"), dpi=120)
    plt.close()
    return result


def check_outliers_zscore(df, columns, threshold=3):
    print(f"\n=== 4. 이상치 (Z-score, |z| > {threshold}) ===")
    rows = []
    for col in columns:
        z = (df[col] - df[col].mean()) / df[col].std()
        count = (z.abs() > threshold).sum()
        rows.append([col, round(df[col].mean(), 3), round(df[col].std(), 3), count])
    result = pd.DataFrame(rows, columns=["컬럼", "평균", "표준편차", "이상치 수"])
    print(result.to_string(index=False))


def show_distributions(df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4))
    for ax, col in zip(axes, columns):
        sns.histplot(df[col], kde=True, ax=ax, color="cornflowerblue")
        ax.set_title(f"{col} 분포")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "04_distributions.png"), dpi=120)
    plt.close()


def check_constant_columns(df):
    print("\n=== 5. 값이 하나뿐인 컬럼 (정보 없음) ===")
    constant = [col for col in df.columns if df[col].nunique() <= 1]
    print(f"개수: {len(constant)}")
    print(constant)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_data(DATA_PATH)

    print("=== 데이터 미리보기 ===")
    print(f"행/열: {df.shape}")
    print(df.head())

    numeric_cols = ["Age", "Fare", "sibsp", "Parch"]

    check_missing(df)
    check_duplicates(df)
    check_outliers_iqr(df, numeric_cols)
    check_outliers_zscore(df, numeric_cols)
    show_distributions(df, numeric_cols)
    check_constant_columns(df)

    print(f"\n그래프 저장 위치: {os.path.abspath(REPORT_DIR)}")


if __name__ == "__main__":
    main()
