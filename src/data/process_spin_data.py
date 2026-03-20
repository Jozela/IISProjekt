import json
import pandas as pd
import os

json_file = "data/raw/lokacija.json"
csv_file = "data/preprocessed/nesrece_v_cestnem_prometu.csv"

# Load JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

events = data.get("value", [])

# Filtriraj nove podatke
new_rows = []
for d in events:
    if d.get("dogodekNaziv") == "Nesreče v cestnem prometu":
        new_rows.append({
            "prijavaCas": d.get("prijavaCas", ""),
            "obcinaNaziv": d.get("obcinaNaziv", ""),
            "nastanekCas": d.get("nastanekCas", "")
        })

new_df = pd.DataFrame(new_rows)

# Če CSV že obstaja → primerjaj
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)

    # združi in odstrani duplikate
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(
        subset=["prijavaCas", "obcinaNaziv", "nastanekCas"]
    )

    added_rows = len(combined_df) - len(existing_df)

    # shrani nazaj
    combined_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"Dodanih novih zapisov: {added_rows}")

else:
    # če CSV še ne obstaja → samo shrani
    new_df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"Ustvarjen CSV z {len(new_df)} zapisi")

print(f"CSV: {csv_file}")