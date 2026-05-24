import os
import shutil
import joblib
import random
import yaml
import subprocess
import sys

import numpy as np
import mlflow
import mlflow.tensorflow
import dagshub
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import NesreceWeatherPreprocessor, SlidingWindowTransformer

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
dagshub.init(
    repo_owner="Jozela",
    repo_name="IISProjekt",
    mlflow=True,
)
mlflow.set_experiment("nesrece_accident_prediction")

# ── Reproducibility ───────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs("models", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_model(input_shape, pos_bias=0.0):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid",
              bias_initializer=tf.keras.initializers.Constant(pos_bias)),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def save_onnx(model, onnx_path):
    saved_model_path = onnx_path.replace(".onnx", "_saved_model")
    model.export(saved_model_path)
    subprocess.run([
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_path,
        "--output", onnx_path,
    ], check=True)
    shutil.rmtree(saved_model_path)


def scale_and_window(df_split, scaler, window_size):
    X_scaled = scaler.transform(df_split[ALL_FEATURES])
    y_labels = df_split[LABEL_COL].values
    combined = np.hstack([X_scaled, y_labels.reshape(-1, 1)])
    X, y     = SlidingWindowTransformer(window_size).fit_transform(combined)
    return X, y.astype(int)


def log_classification_metrics(y_true, y_prob, prefix):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    report = classification_report(
        y_true, y_pred,
        target_names=["No accident", "Accident"],
        zero_division=0,
        output_dict=True,
    )
    mlflow.log_metric(f"{prefix}_precision_accident", report["Accident"]["precision"])
    mlflow.log_metric(f"{prefix}_recall_accident",    report["Accident"]["recall"])
    mlflow.log_metric(f"{prefix}_f1_accident",        report["Accident"]["f1-score"])
    mlflow.log_metric(f"{prefix}_accuracy",           report["accuracy"])

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        mlflow.log_metric(f"{prefix}_roc_auc", auc)
        print(f"{prefix} ROC-AUC: {auc:.4f}")
    else:
        print(f"{prefix} ROC-AUC: skipped — only one class in split")

    print(f"\n── {prefix} classification report (threshold={THRESHOLD}) ──")
    print(classification_report(
        y_true, y_pred,
        target_names=["No accident", "Accident"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


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

if df[LABEL_COL].isna().all():
    print("[ERROR] No labels found in merged data. Exiting.")
    sys.exit(1)

df_model = df[ALL_FEATURES + [LABEL_COL]].copy()

effective_test_size = max(TEST_SIZE, WINDOW_SIZE + 1)
if len(df_model) <= effective_test_size:
    print(f"[ERROR] Not enough data ({len(df_model)} rows). Exiting.")
    sys.exit(1)

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

X_train, y_train = scale_and_window(df_train, preprocess, WINDOW_SIZE)
X_test,  y_test  = scale_and_window(df_test,  preprocess, WINDOW_SIZE)

print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Train accident rate: {y_train.mean():.2%}")
print(f"Test  accident rate: {y_test.mean():.2%}")

neg, pos     = np.bincount(y_train)
raw_ratio    = neg / pos
class_weight = {0: 1.0, 1: float(np.sqrt(raw_ratio))}
print(f"Positive class weight: {class_weight[1]:.1f}x  (raw ratio {raw_ratio:.0f}x)")

# ── MLflow run ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="train_nesrece"):

    mlflow.log_param("nesrece_path",        NESRECE_PATH)
    mlflow.log_param("vreme_path",          VREME_PATH)
    mlflow.log_param("test_size",           TEST_SIZE)
    mlflow.log_param("window_size",         WINDOW_SIZE)
    mlflow.log_param("random_state",        RANDOM_STATE)
    mlflow.log_param("threshold",           THRESHOLD)
    mlflow.log_param("time_freq",           TIME_FREQ)
    mlflow.log_param("class_weight_pos",    round(class_weight[1], 2))
    mlflow.log_param("train_accident_rate", round(float(y_train.mean()), 4))
    mlflow.log_param("n_features",          len(ALL_FEATURES))

    mlflow.tensorflow.autolog()

    early_stopping = EarlyStopping(
        monitor="val_recall", mode="max", patience=7, restore_best_weights=True
    )

    # ── Train on train split ──────────────────────────────────────────────────
    pos_bias = float(np.log(pos / neg))
    model    = build_model((X_train.shape[1], X_train.shape[2]), pos_bias=pos_bias)
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_prob = np.nan_to_num(model.predict(X_test).flatten(), nan=0.0)
    log_classification_metrics(y_test, y_prob, prefix="test")

    # ── Retrain on full dataset ───────────────────────────────────────────────
    preprocess.fit(df_model[ALL_FEATURES])
    X_full, y_full = scale_and_window(df_model, preprocess, WINDOW_SIZE)

    neg_f, pos_f   = np.bincount(y_full.astype(int))
    class_weight_f = {0: 1.0, 1: float(np.sqrt(neg_f / pos_f))}
    pos_bias_f     = float(np.log(pos_f / neg_f))

    model_full = build_model((X_full.shape[1], X_full.shape[2]), pos_bias=pos_bias_f)
    model_full.fit(
        X_full, y_full,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight_f,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_prob_full = np.nan_to_num(model_full.predict(X_full).flatten(), nan=0.0)
    log_classification_metrics(y_full, y_prob_full, prefix="full")

    # ── Save .keras ───────────────────────────────────────────────────────────
    model_path = "models/model_nesrece.keras"
    model_full.save(model_path)
    mlflow.log_artifact(model_path)

    # ── Save ONNX ─────────────────────────────────────────────────────────────
    onnx_path = "models/model_nesrece.onnx"
    try:
        save_onnx(model_full, onnx_path)
        mlflow.log_artifact(onnx_path)
        print(f"ONNX model saved: {onnx_path}")
    except Exception as e:
        print(f"[WARNING] ONNX export failed: {e}")

    # ── Save pipeline & merger ────────────────────────────────────────────────
    pipeline_path = "models/pipeline_nesrece.pkl"
    merger_path   = "models/merger_nesrece.pkl"

    joblib.dump(preprocess, pipeline_path)
    joblib.dump(merger,     merger_path)

    mlflow.log_artifact(pipeline_path)
    mlflow.log_artifact(merger_path)

    print(f"Model saved: {model_path}")
    print(f"Pipeline saved: {pipeline_path}")
    print(f"Merger saved: {merger_path}")