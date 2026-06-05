import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd

from preprocess import NesreceWeatherPreprocessor

params = yaml.safe_load(open("/app/params.yaml"))["train_nesrece"]
NESRECE_PATH = params["nesrece_path"]
VREME_PATH = params["vreme_path"]
WINDOW_SIZE = int(params["window_size"])
TIME_FREQ = params.get("time_freq", "h")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/model_nesrece.pkl")
PIPELINE_PATH = os.environ.get("PIPELINE_PATH", "/app/models/pipeline_nesrece.pkl")
MERGER_PATH = os.environ.get("MERGER_PATH", "/app/models/merger_nesrece.pkl")

OUT_PATH = os.environ.get("PREDICTIONS_PATH", "/app/data/predictions/next24h.json")

NUMERIC_FEATURES = [
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "obcina_enc",
]
BINARY_FEATURES = ["sunny", "rainy", "snowy", "icy", "frost", "fog"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    try:
        merger = joblib.load(MERGER_PATH)
    except Exception:
        merger = NesreceWeatherPreprocessor(time_freq=TIME_FREQ)

    nesrece_df = pd.read_csv(NESRECE_PATH)
    vreme_df = pd.read_csv(VREME_PATH)

    df = merger.fit_transform((nesrece_df, vreme_df))
    df = df.sort_values(["obcinaNaziv", "time_slot"]).reset_index(drop=True)

    # choose "now" as last observed accident hour (not the future weather max)
    nesrece_df["nastanekCas"] = pd.to_datetime(nesrece_df["nastanekCas"], errors="coerce")
    nesrece_df = nesrece_df.dropna(subset=["nastanekCas"])
    last_observed_slot = nesrece_df["nastanekCas"].max().floor(TIME_FREQ)

    future_slots = pd.date_range(
        start=last_observed_slot + pd.Timedelta(hours=1),
        periods=24,
        freq=TIME_FREQ,
    )

    def predict_window(window_df):
        X = pipeline.transform(window_df[ALL_FEATURES])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        x = X.flatten().reshape(1, -1)
        return float(model.predict_proba(x)[0, 1])

    predictions_by_obcina = {}
    top_events = []

    for obcina, g in df.groupby("obcinaNaziv", sort=False):
        g = g.sort_values("time_slot").reset_index(drop=True)

        hist = g[g["time_slot"] <= last_observed_slot]
        if len(hist) < WINDOW_SIZE - 1:
            predictions_by_obcina[obcina] = []
            continue

        hist_tail = hist.tail(WINDOW_SIZE - 1)

        preds = []
        for slot in future_slots:
            hist = g[g["time_slot"] <= last_observed_slot].sort_values("time_slot")

            if len(hist) < WINDOW_SIZE:
                predictions_by_obcina[obcina] = []
                continue

            window = hist.tail(WINDOW_SIZE).copy()

            p = predict_window(window)
            rec = {"time_slot": slot.isoformat(), "probability_percent": round(p * 100, 1)}
            preds.append(rec)

            top_events.append({
                "obcinaNaziv": obcina,
                "time_slot": rec["time_slot"],
                "probability_percent": rec["probability_percent"],
            })

        predictions_by_obcina[obcina] = preds

    top_events = sorted(top_events, key=lambda x: x["probability_percent"], reverse=True)

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "last_observed_slot": last_observed_slot.isoformat(),
        "horizon_hours": 24,
        "top_events": top_events[:200],
        "predictions_by_obcina": predictions_by_obcina,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("Saved:", OUT_PATH)
    if top_events:
        print("Top:", top_events[0])

if __name__ == "__main__":
    main()