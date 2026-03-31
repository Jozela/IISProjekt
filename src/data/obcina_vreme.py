import os
import time
import requests
import pandas as pd
import geopandas as gpd
from datetime import date, timedelta
import yaml
# =========================================================
# CONFIG
# =========================================================
params = yaml.safe_load(open("params.yaml"))["fetch_vreme"]

OBCINE_PATH = params["obcine_path"]
REQUEST_SLEEP = params["request_sleep"]
DAYS_BACK = params["days_back"]
OUTPUT_PATH = "data/raw/obcina_vreme.csv"

# Change this if your municipality name column is different
OBCINA_NAME_COLUMN = "name"

# Open-Meteo archive API
BASE_URL = params["url"]

# Last 2 days
TODAY = date.today()
START_DATE = TODAY - timedelta(days=2)
END_DATE = TODAY - timedelta(days=1)

# Optional: slow down requests a bit
REQUEST_SLEEP = 0.2


# =========================================================
# HELPERS
# =========================================================

def classify_weather(row):
    """
    Create the weather flags you asked for:
    avg temp, rainy, snowy, icy, frost, fog, sunny
    """

    avg_temp = row.get("temperature_2m_mean", None)
    min_temp = row.get("temperature_2m_min", None)
    max_temp = row.get("temperature_2m_max", None)
    rain_sum = row.get("rain_sum", 0) or 0
    snowfall_sum = row.get("snowfall_sum", 0) or 0
    cloud_cover = row.get("cloud_cover_mean", None)
    visibility = row.get("visibility_mean", None)
    sunshine = row.get("sunshine_duration", 0) or 0

    # Flags
    rainy = 1 if rain_sum >= 0.1 else 0
    snowy = 1 if snowfall_sum > 0 else 0
    icy = 1 if pd.notna(max_temp) and max_temp < 0 else 0
    frost = 1 if pd.notna(min_temp) and min_temp < 0 else 0

    # Fog / megla:
    # If visibility is low (< 1000m), treat as fog
    fog = 1 if pd.notna(visibility) and visibility < 1000 else 0

    # Sunny:
    # either sunshine duration is high OR cloud cover low
    sunny = 0
    if pd.notna(sunshine) and sunshine >= 6 * 3600:  # 6 hours in seconds
        sunny = 1
    elif pd.notna(cloud_cover) and cloud_cover < 20 and rainy == 0 and fog == 0:
        sunny = 1

    # Label
    if snowy == 1:
        label = "snowy"
    elif rainy == 1:
        label = "rainy"
    elif fog == 1:
        label = "foggy"
    elif icy == 1:
        label = "icy"
    elif sunny == 1:
        label = "sunny"
    else:
        label = "cloudy_or_mixed"

    return pd.Series({
        "avg_temp_c": avg_temp,
        "min_temp_c": min_temp,
        "max_temp_c": max_temp,
        "precip_mm": rain_sum,
        "snowfall_cm": snowfall_sum,
        "cloud_cover_pct": cloud_cover,
        "visibility_m": visibility,
        "sunshine_duration_sec": sunshine,
        "rainy": rainy,
        "snowy": snowy,
        "icy": icy,
        "frost": frost,
        "fog": fog,
        "sunny": sunny,
        "weather_label": label
    })


def fetch_weather_for_point(lat, lon, start_date, end_date):
    """
    Fetch last 2 days daily weather for a coordinate.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "rain_sum",
            "snowfall_sum",
            "cloud_cover_mean",
            "visibility_mean",
            "sunshine_duration"
        ]),
        "timezone": "Europe/Ljubljana"
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        return pd.DataFrame()

    daily = pd.DataFrame(data["daily"])
    return daily


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading občine GeoJSON...")
    obcine = gpd.read_file(OBCINE_PATH)

    print("Columns found in občine file:")
    print(obcine.columns.tolist())

    if OBCINA_NAME_COLUMN not in obcine.columns:
        raise ValueError(
            f"Column '{OBCINA_NAME_COLUMN}' not found.\n"
            f"Available columns: {obcine.columns.tolist()}"
        )

    # Ensure CRS
    if obcine.crs is None:
        obcine = obcine.set_crs("EPSG:4326")
    else:
        obcine = obcine.to_crs("EPSG:4326")

    # Use centroid of each občina
    obcine["centroid"] = obcine.geometry.centroid
    obcine["lat"] = obcine["centroid"].y
    obcine["lon"] = obcine["centroid"].x

    results = []

    print(f"\nFetching weather for all občine from {START_DATE} to {END_DATE}...\n")

    for idx, row in obcine.iterrows():
        obcina = row[OBCINA_NAME_COLUMN]
        lat = row["lat"]
        lon = row["lon"]

        print(f"[{idx+1}/{len(obcine)}] {obcina}")

        try:
            daily_df = fetch_weather_for_point(lat, lon, START_DATE, END_DATE)

            if daily_df.empty:
                continue

            daily_df["obcina"] = obcina
            daily_df["lat"] = lat
            daily_df["lon"] = lon

            classified = daily_df.apply(classify_weather, axis=1)
            final = pd.concat([daily_df, classified], axis=1)

            results.append(final)

            time.sleep(REQUEST_SLEEP)

        except Exception as e:
            print(f"Failed for {obcina}: {e}")

    if not results:
        raise RuntimeError("No weather data collected.")

    final_df = pd.concat(results, ignore_index=True)

    # Clean final columns
    keep_cols = [
        "time",
        "obcina",
        "lat",
        "lon",
        "avg_temp_c",
        "min_temp_c",
        "max_temp_c",
        "precip_mm",
        "snowfall_cm",
        "cloud_cover_pct",
        "visibility_m",
        "sunshine_duration_sec",
        "sunny",
        "rainy",
        "snowy",
        "icy",
        "frost",
        "fog",
        "weather_label"
    ]
    final_df = final_df[keep_cols].copy()
    final_df = final_df.rename(columns={"time": "date"})

    # Round numeric columns
    round_cols = [
        "lat", "lon",
        "avg_temp_c", "min_temp_c", "max_temp_c",
        "precip_mm", "snowfall_cm",
        "cloud_cover_pct", "visibility_m"
    ]
    for col in round_cols:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").round(2)

    final_df = final_df.sort_values(["date", "obcina"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(final_df.head(20))


if __name__ == "__main__":
    main()