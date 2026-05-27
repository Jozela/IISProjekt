import os
import joblib
import random
import yaml
import json
import sys

import numpy as np
import mlflow
import dagshub
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import shap

from preprocess import NesreceWeatherPreprocessor

# ── Config ────────────────────────────────────────────────────────────────────
params = yaml.safe_load(open("params.yaml"))["train_nesrece"]

NESRECE_PATH = params["nesrece_path"]
VREME_PATH   = params["vreme_path"]
TEST_SIZE    = params["test_size"]
WINDOW_SIZE  = params["window_size"]
RANDOM_STATE = params["random_state"]
THRESHOLD    = params.get("threshold", 0.3)
TIME_FREQ    = params.get("time_freq", "D")

# ── DagsHub + MLflow init ─────────────────────────────────────────────────────
dagshub.init(repo_owner="Jozela", repo_name="IISProjekt", mlflow=True)
mlflow.set_experiment("nesrece_accident_prediction")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs("models", exist_ok=True)

# ── Feature columns ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "day_of_week", "month", "is_weekend",
    "obcina_enc",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES    = NUMERIC_FEATURES + BINARY_FEATURES
LABEL_COL       = "label"

# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading and merging data...")
nesrece_df = pd.read_csv(NESRECE_PATH)
vreme_df   = pd.read_csv(VREME_PATH)

merger = NesreceWeatherPreprocessor(time_freq=TIME_FREQ)
df     = merger.fit_transform((nesrece_df, vreme_df))

print(f"Grid shape: {df.shape}")
print(f"Accident rate: {df['label'].mean():.2%}")

df_model = df[ALL_FEATURES + [LABEL_COL, "obcinaNaziv"]].copy()

effective_test_size = max(TEST_SIZE, WINDOW_SIZE + 1)
df_train = df_model.iloc[:-effective_test_size]
df_test  = df_model.iloc[-effective_test_size:]

# ── Scaler ────────────────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("impute",    SimpleImputer(strategy="mean")),
    ("normalize", MinMaxScaler()),
])
preprocess = ColumnTransformer([
    ("numeric", numeric_transformer, NUMERIC_FEATURES),
    ("binary",  "passthrough",       BINARY_FEATURES),
], remainder="drop")

preprocess.fit(df_train[ALL_FEATURES])

X_train_scaled = preprocess.transform(df_train[ALL_FEATURES])
X_test_scaled  = preprocess.transform(df_test[ALL_FEATURES])
y_train        = df_train[LABEL_COL].values
y_test         = df_test[LABEL_COL].values

# ── Also build lag features (replaces sliding window for XGBoost) ─────────────
def add_lag_features(df_scaled, labels, window_size):
    """Flatten last window_size rows into a single feature vector per sample."""
    X, y = [], []
    for i in range(window_size, len(df_scaled)):
        window = df_scaled[i - window_size:i].flatten()
        X.append(window)
        y.append(labels[i])
    return np.array(X), np.array(y)

X_train_lag, y_train_lag = add_lag_features(X_train_scaled, y_train, WINDOW_SIZE)
X_test_lag,  y_test_lag  = add_lag_features(X_test_scaled,  y_test,  WINDOW_SIZE)

print(f"X_train: {X_train_lag.shape}  positives: {y_train_lag.sum()}")
print(f"X_test:  {X_test_lag.shape}   positives: {y_test_lag.sum()}")

neg, pos   = np.bincount(y_train_lag)
scale_pos  = neg / pos
print(f"scale_pos_weight: {scale_pos:.1f}")

# ── MLflow run ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="train_nesrece"):

    mlflow.log_param("model_type",          "xgboost")
    mlflow.log_param("nesrece_path",        NESRECE_PATH)
    mlflow.log_param("vreme_path",          VREME_PATH)
    mlflow.log_param("test_size",           TEST_SIZE)
    mlflow.log_param("window_size",         WINDOW_SIZE)
    mlflow.log_param("random_state",        RANDOM_STATE)
    mlflow.log_param("threshold",           THRESHOLD)
    mlflow.log_param("time_freq",           TIME_FREQ)
    mlflow.log_param("scale_pos_weight",    round(scale_pos, 2))
    mlflow.log_param("train_accident_rate", round(float(y_train_lag.mean()), 4))

    # ── Train ─────────────────────────────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        early_stopping_rounds=20,
        verbosity=1,
    )
    model.fit(
        X_train_lag, y_train_lag,
        eval_set=[(X_test_lag, y_test_lag)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test_lag)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    report = classification_report(
        y_test_lag, y_pred,
        target_names=["No accident", "Accident"],
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        y_test_lag, y_pred,
        target_names=["No accident", "Accident"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_lag, y_pred))

    mlflow.log_metric("test_precision_accident", report["Accident"]["precision"])
    mlflow.log_metric("test_recall_accident",    report["Accident"]["recall"])
    mlflow.log_metric("test_f1_accident",        report["Accident"]["f1-score"])
    mlflow.log_metric("test_accuracy",           report["accuracy"])

    if len(np.unique(y_test_lag)) > 1:
        auc = roc_auc_score(y_test_lag, y_prob)
        mlflow.log_metric("test_roc_auc", auc)
        print(f"ROC-AUC: {auc:.4f}")

    # ── Retrain on full data ───────────────────────────────────────────────────
    preprocess.fit(df_model[ALL_FEATURES])
    X_full_scaled = preprocess.transform(df_model[ALL_FEATURES])
    y_full        = df_model[LABEL_COL].values
    X_full_lag, y_full_lag = add_lag_features(X_full_scaled, y_full, WINDOW_SIZE)

    neg_f, pos_f  = np.bincount(y_full_lag)
    model_full = XGBClassifier(
        n_estimators=model.best_iteration + 1,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=neg_f / pos_f,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=1,
    )
    model_full.fit(X_full_lag, y_full_lag)

    y_prob_full = model_full.predict_proba(X_full_lag)[:, 1]
    y_pred_full = (y_prob_full >= THRESHOLD).astype(int)
    report_full = classification_report(
        y_full_lag, y_pred_full,
        target_names=["No accident", "Accident"],
        zero_division=0,
        output_dict=True,
    )
    mlflow.log_metric("full_precision_accident", report_full["Accident"]["precision"])
    mlflow.log_metric("full_recall_accident",    report_full["Accident"]["recall"])
    mlflow.log_metric("full_f1_accident",        report_full["Accident"]["f1-score"])
    mlflow.log_metric("full_accuracy",           report_full["accuracy"])
    if len(np.unique(y_full_lag)) > 1:
        mlflow.log_metric("full_roc_auc", roc_auc_score(y_full_lag, y_prob_full))

    # ── Save model & pipeline ─────────────────────────────────────────────────
    model_path    = "models/model_nesrece.pkl"
    pipeline_path = "models/pipeline_nesrece.pkl"
    merger_path   = "models/merger_nesrece.pkl"

    joblib.dump(model_full, model_path)
    joblib.dump(preprocess, pipeline_path)
    joblib.dump(merger,     merger_path)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(pipeline_path)
    mlflow.log_artifact(merger_path)

    print(f"Model saved: {model_path}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("Computing SHAP values...")
    shap_sample_idx = np.random.choice(len(X_full_lag), size=min(500, len(X_full_lag)), replace=False)
    shap_sample     = X_full_lag[shap_sample_idx]

    explainer   = shap.Explainer(model_full.predict_proba, shap_sample)
    shap_values = explainer(shap_sample).values

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # positive class


    # shap_values shape: (n_samples, window*n_features)
    n_feat    = len(ALL_FEATURES)
    shap_3d   = shap_values.reshape(shap_values.shape[0], WINDOW_SIZE, n_feat)
    shap_mean = np.abs(shap_3d).mean(axis=1)

    feature_importance = pd.DataFrame({
        "feature":    ALL_FEATURES,
        "importance": shap_mean.mean(axis=0),
    }).sort_values("importance", ascending=False)
    print(feature_importance.to_string(index=False))

    shap_path = "models/shap_values.npz"
    np.savez(shap_path, shap_values=shap_mean, feature_names=np.array(ALL_FEATURES))
    mlflow.log_artifact(shap_path)

    shap_sample_df = df_model.iloc[
        np.random.choice(len(df_model), size=min(500, len(df_model)), replace=False)
    ].reset_index(drop=True)

    X_shap_scaled = preprocess.transform(shap_sample_df[ALL_FEATURES])
    X_shap_lag, _ = add_lag_features(
        X_shap_scaled, shap_sample_df[LABEL_COL].values, WINDOW_SIZE
    )
    shap_values_obcina = explainer(X_shap_lag).values
    if shap_values_obcina.ndim == 3:
        shap_values_obcina = shap_values_obcina[:, :, 1]

    shap_3d_obcina   = shap_values_obcina.reshape(shap_values_obcina.shape[0], WINDOW_SIZE, n_feat)
    shap_mean_obcina = np.abs(shap_3d_obcina).mean(axis=1)

    obcina_shap    = {}
    shap_obcina_df = shap_sample_df.iloc[WINDOW_SIZE:].reset_index(drop=True)
    for obcina in df_model["obcinaNaziv"].unique():
        mask = shap_obcina_df["obcinaNaziv"] == obcina
        if mask.sum() > 0:
            obcina_shap[obcina] = np.nan_to_num(
                shap_mean_obcina[mask.values].mean(axis=0)
            ).tolist()

    shap_obcina_path = "models/shap_per_obcina.json"
    with open(shap_obcina_path, "w") as f:
        json.dump({
            "feature_names":     ALL_FEATURES,
            "obcina_shap":       obcina_shap,
            "global_importance": feature_importance.set_index("feature")["importance"].to_dict(),
        }, f)
    mlflow.log_artifact(shap_obcina_path)
    print(f"SHAP saved: {shap_path}, {shap_obcina_path}")