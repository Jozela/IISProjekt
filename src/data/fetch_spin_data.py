import requests
import os
import yaml
import json

# Load parameters
params = yaml.safe_load(open("params.yaml"))["fetch"]
json_url = params["url"]

# Ensure folder exists
os.makedirs("data/raw", exist_ok=True)

# Fetch the data
response = requests.get(json_url)
response.raise_for_status()  # check for download errors

# Parse JSON safely
try:
    data = response.json()  # <-- this converts response text into a Python object
except ValueError as e:
    raise ValueError(f"Response is not valid JSON:\n{response.text[:500]}") from e

# Write JSON to file with proper formatting
file_path = "data/raw/lokacija.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Datoteka je uspešno shranjena kot {file_path}")