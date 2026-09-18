from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

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
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
X = df[COLUMNS[:-1]].copy()
X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"학습 {X_train.shape}, 테스트 {X_test.shape}, "
      f"학습 양성비율 {y_train.mean():.2%}, 테스트 양성비율 {y_test.mean():.2%}")

log_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])
rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\n" + "=" * 60)
print("5-fold 교차검증 ROC-AUC (학습 데이터)")
print("=" * 60)
models = {"LogisticRegression": log_reg, "RandomForest": rf}
cv_auc = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_auc[name] = scores.mean()
    print(f"{name:<20} AUC {scores.mean():.3f} (+/- {scores.std():.3f})")

best_name = max(cv_auc, key=cv_auc.get)
best = models[best_name]
print(f"\n최종 모델: {best_name}")

best.fit(X_train, y_train)
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("테스트 세트 성능")
print("=" * 60)
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
}
for k, v in metrics.items():
    print(f"{k:<10} {v:.3f}")
print("\n" + classification_report(y_test, y_pred, target_names=["비당뇨(0)", "당뇨(1)"]))

# ---------------------------------------------------------------
# 시각화 1: 혼동행렬 + ROC + PR 곡선
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["비당뇨", "당뇨"], yticklabels=["비당뇨", "당뇨"])
axes[0].set_title(f"혼동행렬 ({best_name})")
axes[0].set_xlabel("예측"); axes[0].set_ylabel("실제")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="crimson", label=f"AUC = {metrics['ROC-AUC']:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", label="무작위")
axes[1].set_title("ROC 곡선"); axes[1].set_xlabel("위양성률"); axes[1].set_ylabel("진양성률")
axes[1].legend()

prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec, prec, color="steelblue", label=f"PR (양성비율 {y_test.mean():.2f})")
axes[2].set_title("Precision-Recall 곡선"); axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].legend()

plt.tight_layout()
plt.savefig(REPORT_DIR / "08_model_evaluation.png", dpi=150)
print(f"\n저장 완료: {REPORT_DIR / '08_model_evaluation.png'}")

# ---------------------------------------------------------------
# 시각화 2: 변수 중요도 (permutation importance)
# ---------------------------------------------------------------
perm = permutation_importance(best, X_test, y_test, n_repeats=20,
                              random_state=RANDOM_STATE, scoring="roc_auc")
imp = pd.DataFrame({
    "변수": X_test.columns,
    "중요도": perm.importances_mean,
    "표준편차": perm.importances_std,
}).sort_values("중요도", ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(imp["변수"], imp["중요도"], xerr=imp["표준편차"], color="teal")
plt.title(f"변수 중요도 (Permutation, ROC-AUC 기준, {best_name})")
plt.xlabel("AUC 감소량")
plt.tight_layout()
plt.savefig(REPORT_DIR / "09_feature_importance.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '09_feature_importance.png'}")

# ---------------------------------------------------------------
# 시각화 3: 로지스틱 회귀 계수 (해석용)
# ---------------------------------------------------------------
log_reg.fit(X_train, y_train)
coefs = log_reg.named_steps["clf"].coef_[0]
feature_names = log_reg.named_steps["imputer"].get_feature_names_out(X_train.columns)
coef_df = pd.DataFrame({"변수": feature_names, "계수": coefs}).sort_values("계수")

plt.figure(figsize=(9, 6))
colors = ["crimson" if c > 0 else "steelblue" for c in coef_df["계수"]]
plt.barh(coef_df["변수"], coef_df["계수"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("로지스틱 회귀 표준화 계수 (양수=당뇨 위험 증가)")
plt.xlabel("계수 (표준화 단위)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "10_logreg_coefficients.png", dpi=150)
print(f"저장 완료: {REPORT_DIR / '10_logreg_coefficients.png'}")

joblib.dump(best, REPORT_DIR / "diabetes_model.joblib")
print(f"모델 저장 완료: {REPORT_DIR / 'diabetes_model.joblib'}")

plt.show()
