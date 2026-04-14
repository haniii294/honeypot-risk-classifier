import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("=== LOADING DATASET ===")

df = pd.read_csv(
    "../datasets/hp_upnvj_1.honeypots2_500000data.csv",
    sep=";",
    engine="python",
    on_bad_lines="skip"
)

df.columns = [c.lower().strip() for c in df.columns]

required_cols = [
    "fields.source_address",
    "fields.login",
    "fields.target_port"
]

for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Kolom tidak ditemukan: {col}")

df = df[required_cols]

# ==========================================
# 2. DATA CLEANING
# ==========================================
df = df.dropna(subset=required_cols)
df["fields.login"] = df["fields.login"].str.lower()

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("=== FEATURE ENGINEERING ===")

features = df.groupby("fields.source_address").agg(
    connection_count=("fields.login", "count"),
    failed=("fields.login", lambda x: (x == "fail").sum()),
    success=("fields.login", lambda x: (x == "success").sum()),
    unique_ports=("fields.target_port", "nunique")
).reset_index()

features["failed_ratio"] = features["failed"] / features["connection_count"]
features["success_ratio"] = features["success"] / features["connection_count"]

features.fillna(0, inplace=True)

# Filter data invalid
features = features[features["connection_count"] > 0]

print("\nJumlah data setelah agregasi :", len(features))
print("Kolom fitur:", features.columns)

# ==========================================
# 4. LABELING
# ==========================================
print("\n=== LABELING ===")

y = (features["failed_ratio"] > 0.3).astype(int)

X = features[
    [
        "connection_count",
        "failed",
        "success",
        "failed_ratio",
        "success_ratio",
        "unique_ports"
    ]
]

print("\nDistribusi kelas sebelum SMOTE:")
print(y.value_counts())

# ==========================================
# 5. STRATIFIED K-FOLD + SMOTE
# ==========================================
print("\n=== TRAINING WITH STRATIFIED K-FOLD + SMOTE ===")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

all_y_test = []
all_preds = []

fold_number = 1

for train_idx, test_idx in kf.split(X, y):

    print(f"\n--- Fold {fold_number} ---")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # SMOTE hanya di training
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Distribusi setelah SMOTE:")
    print(pd.Series(y_train_res).value_counts())

    # Model baru tiap fold
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X_train_res, y_train_res)

    preds = rf.predict(X_test)

    # Metrics aman
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    acc_scores.append(acc)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

    all_y_test.extend(y_test)
    all_preds.extend(preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    fold_number += 1

# ==========================================
# 6. AVERAGE PERFORMANCE
# ==========================================
print("\n=== AVERAGE PERFORMANCE (5-FOLD) ===")

print(f"Average Accuracy  : {np.mean(acc_scores):.4f}")
print(f"Average Precision : {np.mean(precision_scores):.4f}")
print(f"Average Recall    : {np.mean(recall_scores):.4f}")
print(f"Average F1-score  : {np.mean(f1_scores):.4f}")

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================
print("\n=== CONFUSION MATRIX ===")

cm = confusion_matrix(all_y_test, all_preds)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Risk", "High Risk"],
    yticklabels=["Low Risk", "High Risk"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(all_y_test, all_preds))

# ==========================================
# 8. FEATURE IMPORTANCE (FINAL MODEL NANTI)
# ==========================================

# ==========================================
# 9. TRAIN FINAL MODEL (PENTING!)
# ==========================================
print("\n=== TRAIN FINAL MODEL ===")

smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X, y)

final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(X_res, y_res)

joblib.dump(final_model, "../backend/model/rf_model.pkl")

print("MODEL FINAL DISIMPAN!")

# ==========================================
# 10. FEATURE IMPORTANCE FINAL
# ==========================================
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# ==========================================
# 11. DISTRIBUSI RISIKO
# ==========================================
print("\n=== DISTRIBUSI RISIKO ===")

risk_counts = pd.Series(all_preds).value_counts().sort_index()

print(risk_counts)

risk_counts.index = ["Low Risk", "High Risk"]

plt.figure(figsize=(6,4))
risk_counts.plot(kind="bar")

plt.title("Distribusi Risiko")
plt.xlabel("Kategori")
plt.ylabel("Jumlah")

plt.tight_layout()
plt.savefig("risk_distribution.png", dpi=300)
plt.show()