import requests
import os
import yaml
# Neposredni URL do lokacija.json
params = yaml.safe_load(open("params.yaml"))["fetch"]
# URL to fetch the XML data
json_url = params["url"]
# Ustvari mapo data/raw, če še ne obstaja
os.makedirs("data/raw", exist_ok=True)

# Prenesi JSON datoteko
response = requests.get(json_url)
response.raise_for_status()  # preveri, če je prenos uspešen

# Shrani datoteko v data/raw
file_path = "data/raw/lokacija.json"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(response.text)

print(f"Datoteka je uspešno shranjena kot {file_path}")