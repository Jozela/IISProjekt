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

NUMERIC_FEATURES = [
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "day_of_week", "month", "is_weekend",
    "obcina_enc",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES    = NUMERIC_FEATURES + BINARY_FEATURES

model    = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

nesrece_df = pd.read_csv(NESRECE_PATH)
vreme_df   = pd.read_csv(VREME_PATH)

merger   = NesreceWeatherPreprocessor(time_freq=TIME_FREQ)
df       = merger.fit_transform((nesrece_df, vreme_df))
df_model = df[ALL_FEATURES + ["obcinaNaziv"]].copy()

predictions = {}
for obcina in df_model["obcinaNaziv"].unique():
    obcina_df = df_model[df_model["obcinaNaziv"] == obcina].tail(WINDOW_SIZE)
    if len(obcina_df) < WINDOW_SIZE:
        predictions[obcina] = None
        continue

    X_scaled = pipeline.transform(obcina_df[ALL_FEATURES])
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_input  = X_scaled.flatten().reshape(1, -1)  # flatten window for XGBoost

    prob = float(model.predict_proba(X_input)[0][1])
    if np.isnan(prob) or np.isinf(prob):
        prob = 0.0
    predictions[obcina] = round(prob * 100, 1)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
output = {"date": datetime.today().strftime("%Y-%m-%d"), "predictions": predictions}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

valid = [v for v in predictions.values() if v is not None]
print(f"Predictions saved to {OUTPUT_PATH}")
print(f"Date: {output['date']}")
print(f"Municipalities: {len(predictions)}")
print(f"Avg probability: {np.nanmean(valid):.2f}%")
print(f"Max probability: {max(valid):.2f}%  ({max(predictions, key=lambda k: predictions[k] or 0)})")