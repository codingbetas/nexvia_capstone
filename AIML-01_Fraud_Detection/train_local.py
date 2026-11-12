"""
AIML-01: Real-Time Financial Fraud Detection Platform
Training Pipeline: Feature Engineering, XGBoost with Class Weights, PR-AUC Evaluation, and MLflow Tracking
Dataset: Credit Card Fraud Detection (284,807 transactions, ~0.172% fraud)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, classification_report, average_precision_score

# ── 1. Load Dataset ───────────────────────────────────────────
DATA_PATH = "creditcard.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please provide creditcard.csv.")

print("Loading Credit Card Fraud dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape} | Fraud cases: {df['Class'].sum()} ({df['Class'].mean() * 100:.3f}%)")

# ── 2. Feature Engineering (PRD Step 2) ────────────────────────
print("Engineering domain features: velocity, amount z-scores, and temporal signals...")
df = df.sort_values("Time").reset_index(drop=True)

# Feature 1: Transaction velocity (rolling transaction volume in time windows)
df["tx_velocity_100"] = df["Time"].rolling(window=100, min_periods=1).count()
df["tx_velocity_500"] = df["Time"].rolling(window=500, min_periods=1).count()

# Feature 2: Amount vs user average (expanding z-score as proxy for user spending profile)
amount_mean = df["Amount"].expanding().mean()
amount_std = df["Amount"].expanding().std().fillna(1.0)
df["amount_zscore"] = (df["Amount"] - amount_mean) / np.maximum(amount_std, 1e-6)

# Feature 3: Amount deviation from short-term rolling mean (detects behavioral spikes)
df["amount_rolling_mean"] = df["Amount"].rolling(window=50, min_periods=1).mean()
df["amount_deviation"] = np.abs(df["Amount"] - df["amount_rolling_mean"])

# Feature 4: Log-transformed amount (handles extreme positive skew)
df["log_amount"] = np.log1p(df["Amount"])

# Feature 5: Cyclical temporal encodings (hour of day derived from seconds)
df["hour"] = (df["Time"] % 86400) / 3600.0
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

# Feature 6: Nighttime transaction indicator (higher fraud risk window: 10 PM - 5 AM)
df["is_night"] = ((df["hour"] >= 22.0) | (df["hour"] <= 5.0)).astype(int)

# Compile full feature list
feature_cols = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
    + [
        "tx_velocity_100",
        "tx_velocity_500",
        "amount_zscore",
        "amount_deviation",
        "log_amount",
        "hour_sin",
        "hour_cos",
        "is_night",
    ]
)

print(f"Total features: {len(feature_cols)} (30 original + 8 engineered)")

# ── 3. Train-Test Split (Stratified to preserve imbalance) ────
X = df[feature_cols].fillna(0)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 4. MLflow Experiment Tracking & Training (PRD Step 3) ────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection-platform")

# Calculate class weight ratio for extreme imbalance (~578:1)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = float(neg_count / max(pos_count, 1))

with mlflow.start_run(run_name="xgboost-scale-pos-weight"):
    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "scale_pos_weight": 10.0,  # Balanced penalization for precision-recall tradeoff
        "eval_metric": "aucpr",
        "random_state": 42,
        "tree_method": "hist",
    }
    mlflow.log_params(params)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("imbalance_ratio", f"{scale_weight:.1f}:1")

    print(f"Training XGBoost with scale_pos_weight={params['scale_pos_weight']}...")
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 5. Evaluation: PR-AUC (Not accuracy, per PRD) ─────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    y_pred = (y_pred_proba >= 0.50).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 55)
    print(f"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
    print(f"Average Precision:            {avg_precision:.4f}")
    print(f"Fraud Class Precision:        {report['1']['precision']:.4f}")
    print(f"Fraud Class Recall:           {report['1']['recall']:.4f}")
    print(f"Fraud Class F1-Score:         {report['1']['f1-score']:.4f}")
    print("=" * 55)

    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("average_precision", avg_precision)
    mlflow.log_metric("fraud_precision", report["1"]["precision"])
    mlflow.log_metric("fraud_recall", report["1"]["recall"])
    mlflow.log_metric("fraud_f1", report["1"]["f1-score"])

    # Log model to MLflow
    mlflow.xgboost.log_model(model, "fraud_model")

    # Persist locally for immediate serving
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Success! Model saved to 'fraud_model.pkl' and logged to MLflow.")
