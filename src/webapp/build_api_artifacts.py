# src/app/build_api_artifacts.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from src.models.preprocess import NesreceWeatherPreprocessor

MODEL_PATH   = os.environ.get("MODEL_PATH",   "models/model_nesrece.pkl")
PIPELINE_PATH= os.environ.get("PIPELINE_PATH","models/pipeline_nesrece.pkl")
MERGER_PATH  = os.environ.get("MERGER_PATH",  "models/merger_nesrece.pkl")
SHAP_NPZ     = os.environ.get("SHAP_NPZ",     "models/shap_values.npz")

OUT_TODAY    = os.environ.get("PREDICTIONS_PATH", "data/predictions/today.json")
OUT_HOURLY   = os.environ.get("HOURLY_PATH",      "data/predictions/hourly.json")
OUT_SHAP     = os.environ.get("SHAP_PATH",        "models/shap_per_obcina.json")

# Must match train.py
NUMERIC_FEATURES = [
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "obcina_enc",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "24"))

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    try:
        merger = joblib.load(MERGER_PATH)
    except Exception:
        merger = NesreceWeatherPreprocessor(time_freq="h")

    # Load the same inputs used for training (paths are stored inside the merger usage in your pipeline)
    # We’ll re-read via params.yaml-like convention: easiest is to read the already-preprocessed CSVs.
    nesrece_path = "data/preprocessed/nesrece_v_cestnem_prometu.csv"
    vreme_path   = "data/preprocessed/obcina_vreme_processed.csv"
    nesrece_df = pd.read_csv(nesrece_path)
    vreme_df   = pd.read_csv(vreme_path)

    df = merger.fit_transform((nesrece_df, vreme_df))
    df = df.sort_values(["obcinaNaziv", "time_slot"]).reset_index(drop=True)
    df["time_slot"] = pd.to_datetime(df["time_slot"])

    # Determine "today" date based on latest available slot <= now
    now = pd.Timestamp.now().floor("h")
    last_slot = df.loc[df["time_slot"] <= now, "time_slot"].max()
    if pd.isna(last_slot):
        last_slot = df["time_slot"].max()

    day = last_slot.normalize()
    today_slots = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")

    predictions_daily = {}
    predictions_hourly = {}

    for obcina, g in df.groupby("obcinaNaziv", sort=False):
        g = g.sort_values("time_slot").reset_index(drop=True)

        # hourly probs for today's 24h, if available
        probs = []
        for slot in today_slots:
            # build window ending at this slot
            idx = g.index[g["time_slot"] == slot]
            if len(idx) == 0:
                probs.append(None)
                continue
            i = int(idx[0])
            if i < WINDOW_SIZE:
                probs.append(None)
                continue

            window = g.iloc[i - WINDOW_SIZE:i]
            X = pipeline.transform(window[ALL_FEATURES])
            x = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).flatten().reshape(1, -1)
            p = float(model.predict_proba(x)[0, 1]) * 100.0
            probs.append(round(p, 1))

        # store hourly (frontend wants list length 24)
        predictions_hourly[obcina] = probs

        # daily: choose mean of available hours (ignore None)
        vals = [v for v in probs if v is not None]
        predictions_daily[obcina] = round(float(np.mean(vals)), 1) if vals else None

    save_json(OUT_TODAY, {
        "date": day.date().isoformat(),
        "predictions": predictions_daily,
    })

    save_json(OUT_HOURLY, {
        "date": day.date().isoformat(),
        "predictions": predictions_hourly,
    })

    # SHAP export: global only (API already falls back to this)
    shap_payload = {"feature_names": ALL_FEATURES, "obcina_shap": {}, "global_importance": {}}
    if os.path.exists(SHAP_NPZ):
        data = np.load(SHAP_NPZ, allow_pickle=True)
        shap_values = data["shap_values"]  # shape: (n_samples, n_features)
        # importance per feature = mean absolute shap over samples
        imp = np.abs(shap_values).mean(axis=0)
        shap_payload["global_importance"] = {f: float(v) for f, v in zip(ALL_FEATURES, imp)}
    save_json(OUT_SHAP, shap_payload)

    print("Wrote:")
    print(" -", OUT_TODAY)
    print(" -", OUT_HOURLY)
    print(" -", OUT_SHAP)

if __name__ == "__main__":
    main()