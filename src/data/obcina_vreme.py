import os
import time
import requests
import pandas as pd
import geopandas as gpd
import yaml

params = yaml.safe_load(open("params.yaml"))["fetch_vreme"]

OBCINE_PATH = params["obcine_path"]
REQUEST_SLEEP = float(params.get("request_sleep", 0.2))
OUTPUT_PATH = params.get("output_path", "data/raw/obcina_vreme.csv")

BASE_URL = params["url"]  # MUST be https://api.open-meteo.com/v1/forecast
OBCINA_NAME_COLUMN = params.get("obcina_name_column", "name")
TIMEZONE = params.get("timezone", "Europe/Ljubljana")

PAST_DAYS = int(params.get("past_days", 30))
HORIZON_HOURS = int(params.get("horizon_hours", 48))  # >=24

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "visibility",
    "wind_speed_10m",
]

def classify_weather_hourly(row):
    temp = row.get("temperature_2m", None)
    precip = row.get("precipitation", 0) or 0
    snowfall = row.get("snowfall", 0) or 0
    cloud_cover = row.get("cloud_cover", None)
    visibility = row.get("visibility", None)

    rainy = 1 if precip >= 0.1 else 0
    snowy = 1 if snowfall > 0 else 0
    icy = 1 if pd.notna(temp) and temp < 0 else 0
    frost = 1 if pd.notna(temp) and temp < 0 else 0
    fog = 1 if pd.notna(visibility) and visibility < 1000 else 0
    sunny = 1 if (pd.notna(cloud_cover) and cloud_cover < 20 and rainy == 0 and fog == 0) else 0

    if snowy:
        label = "snowy"
    elif rainy:
        label = "rainy"
    elif fog:
        label = "foggy"
    elif icy:
        label = "icy"
    elif sunny:
        label = "sunny"
    else:
        label = "cloudy_or_mixed"

    return pd.Series({
        "avg_temp_c": temp,
        "min_temp_c": temp,
        "max_temp_c": temp,
        "precip_mm": precip,
        "snowfall_cm": snowfall,
        "cloud_cover_pct": cloud_cover,
        "visibility_m": visibility,
        "sunshine_duration_sec": 3600 if sunny else 0,
        "rainy": rainy,
        "snowy": snowy,
        "icy": icy,
        "frost": frost,
        "fog": fog,
        "sunny": sunny,
        "weather_label": label,
    })

def fetch_weather_for_point(lat, lon):
    # Explicit date window: (now - past_days) .. (now + horizon_hours)
    now = pd.Timestamp.now(tz=TIMEZONE).floor("h")
    start_date = (now - pd.Timedelta(days=PAST_DAYS)).date().isoformat()
    end_date = (now + pd.Timedelta(hours=HORIZON_HOURS)).date().isoformat()

    q = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
    }

    r = requests.get(BASE_URL, params=q, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    return pd.DataFrame(hourly)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    obcine = gpd.read_file(OBCINE_PATH)
    if OBCINA_NAME_COLUMN not in obcine.columns:
        raise ValueError(
            f"Column '{OBCINA_NAME_COLUMN}' not found. "
            f"Available columns: {obcine.columns.tolist()}"
        )

    if obcine.crs is None:
        obcine = obcine.set_crs("EPSG:4326")
    else:
        obcine = obcine.to_crs("EPSG:4326")

    # NOTE: centroid in geographic CRS is an approximation; OK for this use.
    obcine["centroid"] = obcine.geometry.centroid
    obcine["lat"] = obcine["centroid"].y
    obcine["lon"] = obcine["centroid"].x

    results = []
    for i, row in obcine.reset_index(drop=True).iterrows():
        obcina = row[OBCINA_NAME_COLUMN]
        lat = float(row["lat"])
        lon = float(row["lon"])

        try:
            hourly_df = fetch_weather_for_point(lat, lon)
            if hourly_df.empty:
                continue

            hourly_df["obcina"] = obcina
            hourly_df["lat"] = lat
            hourly_df["lon"] = lon

            classified = hourly_df.apply(classify_weather_hourly, axis=1)
            final = pd.concat([hourly_df, classified], axis=1)
            results.append(final)

            time.sleep(REQUEST_SLEEP)

        except Exception as e:
            print(f"Failed for {obcina}: {e}")

    if not results:
        raise RuntimeError("No weather data collected.")

    final_df = pd.concat(results, ignore_index=True)

    keep_cols = [
        "time", "obcina", "lat", "lon",
        "avg_temp_c", "min_temp_c", "max_temp_c",
        "precip_mm", "snowfall_cm", "cloud_cover_pct",
        "visibility_m", "sunshine_duration_sec",
        "sunny", "rainy", "snowy", "icy", "frost", "fog",
        "weather_label",
    ]
    final_df = final_df[[c for c in keep_cols if c in final_df.columns]].copy()
    final_df = final_df.rename(columns={"time": "date"})

    final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")
    final_df = final_df.dropna(subset=["date"])
    final_df = final_df.drop_duplicates(subset=["obcina", "date"])

    final_df = final_df.sort_values(["date", "obcina"]).reset_index(drop=True)

    # write ISO timestamps with hour
    out = final_df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%dT%H:%M")
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Saved: {OUTPUT_PATH}")
    print("min/max:", out["date"].min(), out["date"].max())

if __name__ == "__main__":
    main()