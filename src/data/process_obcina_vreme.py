import os
import pandas as pd
import os
import yaml
import pandas as pd 
# =========================
# CONFIG
# =========================
params = yaml.safe_load(open("params.yaml"))["preprocess_vreme"]

RAW_PATH = params["raw_path"]
PROCESSED_PATH = params["processed_path"]

PROCESSED_DIR = os.path.dirname(PROCESSED_PATH)

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
backup_file = "/tmp/vreme_backup.csv"
if os.path.exists(backup_file):
    existing_df = pd.read_csv(backup_file)
    combined_df = pd.concat([existing_df, raw_df], ignore_index=True)
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