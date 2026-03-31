import os
import pandas as pd

# =========================
# CONFIG
# =========================

RAW_PATH = "data/raw/obcina_vreme.csv"
PROCESSED_DIR = "data/preprocessed"
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "obcina_vreme_processed.csv")

# =========================
# CREATE FOLDER IF MISSING
# =========================
os.makedirs(PROCESSED_DIR, exist_ok=True)

# =========================
# LOAD RAW DATA
# =========================
raw_df = pd.read_csv(RAW_PATH)

# Remove lat/lon columns if they exist
for col in ["lat", "lon"]:
    if col in raw_df.columns:
        raw_df = raw_df.drop(columns=col)

# =========================
# APPEND TO PROCESSED FILE
# =========================
if os.path.exists(PROCESSED_PATH):
    processed_df = pd.read_csv(PROCESSED_PATH)
    combined_df = pd.concat([processed_df, raw_df], ignore_index=True)
else:
    combined_df = raw_df

# Optional: remove duplicates (same date + obcina)
combined_df = combined_df.drop_duplicates(subset=["date", "obcina"], keep="last")

# =========================
# SAVE BACK
# =========================
combined_df.to_csv(PROCESSED_PATH, index=False)
print(f"Processed data saved to: {PROCESSED_PATH}")
print(f"Total rows now: {len(combined_df)}")