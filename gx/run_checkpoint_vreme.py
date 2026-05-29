import sys
import pandas as pd

PATH = "../data/preprocessed/obcina_vreme_processed.csv"
HORIZON_HOURS = 24

REQUIRED_COLS = [
    "date", "obcina",
    "avg_temp_c", "min_temp_c", "max_temp_c",
    "precip_mm", "snowfall_cm", "cloud_cover_pct",
    "sunshine_duration_sec",
    "sunny", "rainy", "snowy", "icy", "frost", "fog",
]

def main():
    df = pd.read_csv(PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print("Missing required columns:", missing)
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        print("Some weather dates are not parseable.")
        sys.exit(1)

    # uniqueness
    dup = df.duplicated(subset=["obcina", "date"]).sum()
    if dup:
        print(f"Duplicate (obcina, date) rows: {dup}")
        sys.exit(1)

    # require data to extend at least next 24h beyond 'now'
    now = pd.Timestamp.now().floor("H")
    required_max = now + pd.Timedelta(hours=HORIZON_HOURS)

    max_date = df["date"].max()
    min_date = df["date"].min()

    print("Weather date range:", min_date, "->", max_date)
    print("Now:", now, "Required max:", required_max)

    if max_date < required_max:
        print("Weather data does not include enough future hours for next-24h prediction.")
        sys.exit(1)

    # hourly check (optional but strongly recommended for next-24h hourly model)
    # This fails if you only have daily rows at 00:00.
    if df["date"].dt.minute.ne(0).any() or df["date"].dt.second.ne(0).any():
        # fine; still hourly-aligned
        pass

    # quick heuristic: do we have more than 1 unique hour?
    if df["date"].dt.hour.nunique() < 2:
        print("Weather data appears not to be hourly (only one hour-of-day present).")
        sys.exit(1)

    print("validate_vreme: OK")
    sys.exit(0)

if __name__ == "__main__":
    main()