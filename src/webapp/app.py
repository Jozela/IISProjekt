import os
import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PREDICTIONS_PATH = os.environ.get("PREDICTIONS_PATH", "data/predictions/today.json")
HOURLY_PATH      = os.environ.get("HOURLY_PATH",      "data/predictions/hourly.json")
SHAP_PATH        = os.environ.get("SHAP_PATH",        "models/shap_per_obcina.json")

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predictions")
def predictions():
    data = load_json(PREDICTIONS_PATH)
    return jsonify({"status": "ok", **data})

@app.route("/api/predictions/hourly/<obcina>")
def hourly(obcina):
    data  = load_json(HOURLY_PATH)
    preds = data.get("predictions", {})
    hours = preds.get(obcina) or preds.get(
        next((k for k in preds if k.lower() == obcina.lower()), None), None
    )
    return jsonify({
        "status":  "ok",
        "obcina":  obcina,
        "date":    data.get("date"),
        "hours":   list(range(24)),
        "probabilities": hours,
    })

@app.route("/api/shap/<obcina>")
def shap_endpoint(obcina):
    shap_data     = load_json(SHAP_PATH)
    feature_names = shap_data.get("feature_names", [])
    obcina_shap   = shap_data.get("obcina_shap", {})
    global_imp    = shap_data.get("global_importance", {})

    values = obcina_shap.get(obcina) or obcina_shap.get(
        next((k for k in obcina_shap if k.lower() == obcina.lower()), None), None
    )
    if values is None:
        values = [global_imp.get(f, 0) for f in feature_names]

    features = sorted(zip(feature_names, values), key=lambda x: x[1], reverse=True)
    return jsonify({
        "status":   "ok",
        "obcina":   obcina,
        "features": [{"name": n, "importance": round(v, 6)} for n, v in features],
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)