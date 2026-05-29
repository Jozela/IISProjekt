import os
import yaml
import pandas as pd

def main():
    params = yaml.safe_load(open("params.yaml"))["preprocess_vreme"]
    raw_path = params["raw_path"]
    processed_path = params["processed_path"]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    df = pd.read_csv(raw_path)

    # Drop lat/lon if present (optional)
    for col in ("lat", "lon"):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Parse datetime, keep hourly timestamps
    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in {raw_path}. Columns: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "obcina"])

    # Enforce one row per municipality-hour
    df = df.drop_duplicates(subset=["obcina", "date"], keep="last")

    # Sort for reproducibility
    df = df.sort_values(["date", "obcina"]).reset_index(drop=True)

    # Save with ISO timestamps including hour/minute (matches your raw format)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%dT%H:%M")
    out.to_csv(processed_path, index=False, encoding="utf-8")

    print(f"Processed data saved to: {processed_path}")
    print(f"Total rows now: {len(out)}")
    print("Min/max:", out["date"].min(), "->", out["date"].max())
    print("Unique hours:", sorted(pd.to_datetime(out["date"]).dt.hour.unique()))

if __name__ == "__main__":
    main()