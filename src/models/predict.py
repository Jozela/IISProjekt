import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from preprocess import NesreceWeatherPreprocessor

params = yaml.safe_load(open("params.yaml"))["train_nesrece"]

NESRECE_PATH = params["nesrece_path"]
VREME_PATH   = params["vreme_path"]
WINDOW_SIZE  = params["window_size"]
TIME_FREQ    = params.get("time_freq", "D")

MODEL_PATH    = "models/model_nesrece.pkl"
PIPELINE_PATH = "models/pipeline_nesrece.pkl"
OUTPUT_PATH   = "data/predictions/today.json"
HOURLY_PATH   = "data/predictions/hourly.json"

NUMERIC_FEATURES = [
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "day_of_week", "month", "is_weekend",
    "obcina_enc",
    "avg_accident_hour",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES    = NUMERIC_FEATURES + BINARY_FEATURES

model    = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

nesrece_df = pd.read_csv(NESRECE_PATH)
vreme_df   = pd.read_csv(VREME_PATH)

merger   = NesreceWeatherPreprocessor(time_freq=TIME_FREQ)
df       = merger.fit_transform((nesrece_df, vreme_df))
df_model = df[ALL_FEATURES + ["obcinaNaziv", "day_slot"]].copy()

# ── Daily predictions ─────────────────────────────────────────────────────────
daily_predictions = {}
for obcina in df_model["obcinaNaziv"].unique():
    obcina_df = df_model[df_model["obcinaNaziv"] == obcina].tail(WINDOW_SIZE)
    if len(obcina_df) < WINDOW_SIZE:
        daily_predictions[obcina] = None
        continue

    X_scaled = pipeline.transform(obcina_df[ALL_FEATURES])
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_input  = X_scaled.flatten().reshape(1, -1)

    prob = float(model.predict_proba(X_input)[0][1])
    if np.isnan(prob) or np.isinf(prob):
        prob = 0.0
    daily_predictions[obcina] = round(prob * 100, 1)

# ── Hourly predictions ────────────────────────────────────────────────────────
# For each obcina, simulate each hour of today by setting avg_accident_hour
hourly_predictions = {}
HOURS = list(range(24))

for obcina in df_model["obcinaNaziv"].unique():
    obcina_df = df_model[df_model["obcinaNaziv"] == obcina].tail(WINDOW_SIZE).copy()
    if len(obcina_df) < WINDOW_SIZE:
        hourly_predictions[obcina] = None
        continue

    hour_probs = []
    for hour in HOURS:
        # Set avg_accident_hour to this specific hour for the last row
        obcina_hour = obcina_df.copy()
        obcina_hour.iloc[-1, obcina_hour.columns.get_loc("avg_accident_hour")] = hour

        X_scaled = pipeline.transform(obcina_hour[ALL_FEATURES])
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_input  = X_scaled.flatten().reshape(1, -1)

        prob = float(model.predict_proba(X_input)[0][1])
        if np.isnan(prob) or np.isinf(prob):
            prob = 0.0
        hour_probs.append(round(prob * 100, 1))

    hourly_predictions[obcina] = hour_probs  # list of 24 values

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

today = datetime.today().strftime("%Y-%m-%d")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"date": today, "predictions": daily_predictions}, f,
              ensure_ascii=False, indent=2, allow_nan=False)

with open(HOURLY_PATH, "w", encoding="utf-8") as f:
    json.dump({"date": today, "predictions": hourly_predictions}, f,
              ensure_ascii=False, indent=2, allow_nan=False)

valid = [v for v in daily_predictions.values() if v is not None]
print(f"Daily predictions saved to {OUTPUT_PATH}")
print(f"Hourly predictions saved to {HOURLY_PATH}")
print(f"Date: {today}")
print(f"Municipalities: {len(daily_predictions)}")
print(f"Avg probability: {np.nanmean(valid):.2f}%")
print(f"Max probability: {max(valid):.2f}%  ({max(daily_predictions, key=lambda k: daily_predictions[k] or 0)})")