import os
import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

PREDICTIONS_PATH = os.environ.get("PREDICTIONS_PATH", "data/predictions/today.json")
SHAP_PATH        = os.environ.get("SHAP_PATH",        "models/shap_per_obcina.json")

def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return {"date": None, "predictions": {}}
    with open(PREDICTIONS_PATH) as f:
        return json.load(f)

def load_shap():
    if not os.path.exists(SHAP_PATH):
        return {}
    with open(SHAP_PATH) as f:
        return json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predictions")
def predictions():
    data = load_predictions()
    return jsonify({"status": "ok", **data})

@app.route("/api/shap/<obcina>")
def shap_endpoint(obcina):
    shap_data     = load_shap()
    feature_names = shap_data.get("feature_names", [])
    obcina_shap   = shap_data.get("obcina_shap", {})
    global_imp    = shap_data.get("global_importance", {})

    values = obcina_shap.get(obcina) or obcina_shap.get(
        next((k for k in obcina_shap if k.lower() == obcina.lower()), None), None
    )
    if values is None:
        values = [global_imp.get(f, 0) for f in feature_names]

    features = sorted(
        zip(feature_names, values),
        key=lambda x: x[1], reverse=True
    )
    return jsonify({
        "status":   "ok",
        "obcina":   obcina,
        "features": [{"name": n, "importance": round(v, 6)} for n, v in features],
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)