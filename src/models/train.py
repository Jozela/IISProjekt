import os
import random
import json
import joblib
import yaml

import numpy as np
import pandas as pd
import mlflow
import dagshub
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import shap

from preprocess import NesreceWeatherPreprocessor

# ── Config ────────────────────────────────────────────────────────────────────
params = yaml.safe_load(open("params.yaml"))["train_nesrece"]
NESRECE_PATH = params["nesrece_path"]
VREME_PATH = params["vreme_path"]
TEST_SIZE = params["test_size"]
WINDOW_SIZE = params["window_size"]
RANDOM_STATE = params["random_state"]
THRESHOLD = params.get("threshold", 0.3)
TIME_FREQ = params.get("time_freq", "H")

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
    "hour_of_day",
    "day_of_week", "month", "is_weekend",
    "obcina_enc",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
LABEL_COL = "label"

# ── Load & merge ──────────────────────────────────────────────────────────────
print("Loading and merging data...")
nesrece_df = pd.read_csv(NESRECE_PATH)
vreme_df = pd.read_csv(VREME_PATH)

merger = NesreceWeatherPreprocessor(time_freq=TIME_FREQ)
df = merger.fit_transform((nesrece_df, vreme_df))

print(f"Grid shape: {df.shape}")
print(f"Accident rate: {df[LABEL_COL].mean():.2%}")

df_model = df[["obcinaNaziv", "time_slot"] + ALL_FEATURES + [LABEL_COL]].copy()

# Time split: keep last TEST_SIZE rows PER OBČINA (not global tail)
def split_train_test_per_group(df_in, group_col, time_col, test_size):
    parts_train = []
    parts_test = []
    for _, g in df_in.sort_values([group_col, time_col]).groupby(group_col, sort=False):
        if len(g) <= test_size:
            # if too short, put all into train (or handle differently)
            parts_train.append(g)
            continue
        parts_train.append(g.iloc[:-test_size])
        parts_test.append(g.iloc[-test_size:])
    return pd.concat(parts_train, ignore_index=True), pd.concat(parts_test, ignore_index=True)

effective_test_size = max(TEST_SIZE, WINDOW_SIZE + 1)
df_train, df_test = split_train_test_per_group(
    df_model, group_col="obcinaNaziv", time_col="time_slot", test_size=effective_test_size
)

# ── Scaling ───────────────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("normalize", MinMaxScaler()),
])
preprocess = ColumnTransformer([
    ("numeric", numeric_transformer, NUMERIC_FEATURES),
    ("binary", "passthrough", BINARY_FEATURES),
], remainder="drop")

preprocess.fit(df_train[ALL_FEATURES])

# ── Lag windows per obcina (CRITICAL FIX) ─────────────────────────────────────
def make_lagged_Xy(df_in, preprocess, features, label_col, window_size,
                  group_col="obcinaNaziv", time_col="time_slot"):
    X_list, y_list = [], []
    for _, g in df_in.sort_values([group_col, time_col]).groupby(group_col, sort=False):
        X_scaled = preprocess.transform(g[features])
        y = g[label_col].to_numpy()

        if len(g) <= window_size:
            continue

        for i in range(window_size, len(g)):
            X_list.append(X_scaled[i - window_size:i].flatten())
            y_list.append(y[i])

    return np.asarray(X_list), np.asarray(y_list)

X_train_lag, y_train_lag = make_lagged_Xy(df_train, preprocess, ALL_FEATURES, LABEL_COL, WINDOW_SIZE)
X_test_lag, y_test_lag = make_lagged_Xy(df_test, preprocess, ALL_FEATURES, LABEL_COL, WINDOW_SIZE)

print(f"X_train: {X_train_lag.shape}  positives: {y_train_lag.sum()}")
print(f"X_test:  {X_test_lag.shape}  positives: {y_test_lag.sum()}")

neg, pos = np.bincount(y_train_lag)
scale_pos = (neg / max(pos, 1))
print(f"scale_pos_weight: {scale_pos:.1f}")

# ── MLflow run ────────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="train_nesrece"):

    mlflow.log_param("model_type", "xgboost")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("window_size", WINDOW_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("threshold", THRESHOLD)
    mlflow.log_param("time_freq", TIME_FREQ)
    mlflow.log_param("scale_pos_weight", round(float(scale_pos), 2))
    mlflow.log_param("train_accident_rate", round(float(y_train_lag.mean()), 6))

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

    # Evaluate
    y_prob = model.predict_proba(X_test_lag)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print(classification_report(y_test_lag, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_lag, y_pred))

    report = classification_report(y_test_lag, y_pred, zero_division=0, output_dict=True)
    mlflow.log_metric("test_precision_pos", report["1"]["precision"])
    mlflow.log_metric("test_recall_pos", report["1"]["recall"])
    mlflow.log_metric("test_f1_pos", report["1"]["f1-score"])
    mlflow.log_metric("test_accuracy", report["accuracy"])

    if len(np.unique(y_test_lag)) > 1:
        auc = roc_auc_score(y_test_lag, y_prob)
        mlflow.log_metric("test_roc_auc", auc)
        print(f"ROC-AUC: {auc:.4f}")

    # Retrain on full data (all groups)
    preprocess.fit(df_model[ALL_FEATURES])
    X_full_lag, y_full_lag = make_lagged_Xy(df_model, preprocess, ALL_FEATURES, LABEL_COL, WINDOW_SIZE)

    neg_f, pos_f = np.bincount(y_full_lag)
    model_full = XGBClassifier(
        n_estimators=(model.best_iteration + 1) if hasattr(model, "best_iteration") else 300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=(neg_f / max(pos_f, 1)),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=1,
    )
    model_full.fit(X_full_lag, y_full_lag)

    # Save artifacts
    model_path = "models/model_nesrece.pkl"
    pipeline_path = "models/pipeline_nesrece.pkl"
    merger_path = "models/merger_nesrece.pkl"

    joblib.dump(model_full, model_path)
    joblib.dump(preprocess, pipeline_path)
    joblib.dump(merger, merger_path)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(pipeline_path)
    mlflow.log_artifact(merger_path)

    print(f"Model saved: {model_path}")

        # Optional: SHAP (PermutationExplainer needs max_evals >= 2*num_features + 1)
    print("Computing SHAP values...")

    shap_sample_idx = np.random.choice(
        len(X_full_lag),
        size=min(500, len(X_full_lag)),
        replace=False,
    )
    shap_sample = X_full_lag[shap_sample_idx]

    # For lagged windows the "feature" count is window_size * n_features
    n_flat_features = shap_sample.shape[1]
    max_evals = max(500, 2 * n_flat_features + 1)

    # Use a small background set for speed; still must satisfy max_evals constraint
    bg_idx = np.random.choice(
        len(X_full_lag),
        size=min(100, len(X_full_lag)),
        replace=False,
    )
    background = X_full_lag[bg_idx]

    # Explicitly use PermutationExplainer so behavior is predictable
    explainer = shap.PermutationExplainer(model_full.predict_proba, background)
    shap_values = explainer(shap_sample, max_evals=max_evals).values

    # predict_proba returns 2 classes -> take positive class
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    n_feat = len(ALL_FEATURES)
    shap_3d = shap_values.reshape(shap_values.shape[0], WINDOW_SIZE, n_feat)
    shap_mean = np.abs(shap_3d).mean(axis=1)

    feature_importance = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": shap_mean.mean(axis=0),
    }).sort_values("importance", ascending=False)

    print(feature_importance.to_string(index=False))

    shap_path = "models/shap_values.npz"
    np.savez(shap_path, shap_values=shap_mean, feature_names=np.array(ALL_FEATURES))
    mlflow.log_artifact(shap_path)