import os
import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import requests
import numpy as np
import pandas as pd
from flask import request
from chat_route import chat_bp
app = Flask(__name__)
CORS(app)
CENTROIDS_PATH   = os.environ.get("CENTROIDS_PATH", "/app/data/obcine_centroids.json")
PREDICTIONS_PATH = os.environ.get("PREDICTIONS_PATH", "/app/data/predictions/next24h.json")
HOURLY_PATH      = os.environ.get("HOURLY_PATH", "/app/data/predictions/hourly.json")
SHAP_PATH        = os.environ.get("SHAP_PATH", "/app/models/shap_per_obcina.json")
app.register_blueprint(chat_bp)

def load_centroids():
    return load_json(CENTROIDS_PATH) or {}

def nearest_obcina(lat, lon, centroids):
    best_name, best_d = None, 1e18
    for name, (clat, clon) in centroids.items():
        d = (lat - clat) ** 2 + (lon - clon) ** 2
        if d < best_d:
            best_d, best_name = d, name
    return best_name

def nominatim_geocode(q):
    url = "https://nominatim.openstreetmap.org/search"
    r = requests.get(
        url,
        params={"q": q, "format": "json", "limit": 1},
        headers={"User-Agent": "ISSProjekt/1.0"},
        timeout=20,
    )
    r.raise_for_status()
    arr = r.json()
    if not arr:
        return None
    return float(arr[0]["lat"]), float(arr[0]["lon"])

def osrm_route(lat1, lon1, lat2, lon2):
    url = f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js.get("routes"):
        return None
    route = js["routes"][0]
    coords = route["geometry"]["coordinates"]  # list of [lon, lat]
    return coords, float(route["duration"]), float(route["distance"])

def aggregate_probs(ps):
    p_no = 1.0
    for p in ps:
        p_no *= (1.0 - p)
    return 1.0 - p_no

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

from flask import request
import math

def clamp01(x):
    return max(0.0, min(1.0, x))

@app.route("/api/route_risk")
def route_risk():
    start = request.args.get("from", "")
    end   = request.args.get("to", "")
    start_hour = int(request.args.get("start_hour", 8))

    hourly_data = load_json(HOURLY_PATH)
    preds = (hourly_data.get("predictions") or {})

    # MINI: hardcoded "pot" (to si kasneje izboljšaš z routingom)
    # Za demo je dovolj nekaj občin med MB in CE. Prilagodi na svoje ključe v JSON.
    if start.lower().startswith("maribor") and end.lower().startswith("celje"):
        route_obcine = ["Maribor", "Slovenska Bistrica", "Slovenske Konjice", "Celje"]
    else:
        # fallback: samo start + end
        route_obcine = [start, end]

    # vzamemo 1 uro (ali lahko več ur za daljšo pot)
    hours = [start_hour]  # npr. [start_hour, start_hour+1] če hočeš 2h okno

    # zberemo p za vsak segment
    segment_probs = []
    for obcina in route_obcine:
        hours_list = preds.get(obcina) or preds.get(
            next((k for k in preds if k.lower() == obcina.lower()), None), None
        )

        # če ni podatkov, preskoči (ali nastavi None)
        if not hours_list:
            continue

        for h in hours:
            if 0 <= h < len(hours_list):
                p_pct = hours_list[h]
                if p_pct is None:
                    continue
                segment_probs.append(clamp01(float(p_pct) / 100.0))

    # agregacija: 1 - Π(1-p)
    p_no = 1.0
    for p in segment_probs:
        p_no *= (1.0 - p)

    p_any = 1.0 - p_no

    return jsonify({
        "status": "ok",
        "from": start,
        "to": end,
        "date": hourly_data.get("date"),
        "start_hour": start_hour,
        "route_obcine": route_obcine,
        "segments_used": len(segment_probs),
        "risk_probability": round(p_any * 100.0, 2),
        "note": "Agregirano iz občinskih urnih napovedi (ni osebno tveganje, ni kalibrirano)."
    })
@app.route("/api/route_risk_live", methods=["POST"])
def route_risk_live():
    body = request.get_json(force=True) or {}

    # frontend pošilja: start_lat/start_lon/dest_lat/dest_lon
    try:
        start_lat = float(body["start_lat"])
        start_lon = float(body["start_lon"])
        dest_lat  = float(body["dest_lat"])
        dest_lon  = float(body["dest_lon"])
    except Exception:
        return jsonify({"status": "error", "message": "Missing start_lat/start_lon/dest_lat/dest_lon"}), 400

    # 1) routing (OSRM)
    try:
        out = osrm_route(start_lat, start_lon, dest_lat, dest_lon)  # mora vrnit (coords, duration_s, distance_m)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Routing failed: {e}"}), 502

    if out is None:
        return jsonify({"status": "error", "message": "Routing failed"}), 502

    coords, duration_s, distance_m = out  # coords: list of [lon, lat]

    # 2) podatki napovedi + centroidi
    hourly_data = load_json(HOURLY_PATH)
    preds = (hourly_data.get("predictions") or {})
    centroids = load_centroids()
    if not centroids:
        return jsonify({"status": "error", "message": "Missing centroids (data/obcine_centroids.json)"}), 500

    # 3) naredi route točke + p% za trenutno uro (za barvanje)
    import pandas as pd  # ok tudi, če je na vrhu; tu je samo da dela
    now_hour = int(pd.Timestamp.now().hour)

    route_points = []
    step = max(1, len(coords) // 80)  # največ ~80 točk, da ni preveč
    for (lon, lat) in coords[::step]:
        obcina = nearest_obcina(lat, lon, centroids)

        p_pct = 0.0
        if obcina:
            hours_list = preds.get(obcina) or preds.get(
                next((k for k in preds if k.lower() == obcina.lower()), None), None
            )
            if hours_list and 0 <= now_hour < len(hours_list):
                p = hours_list[now_hour]
                if p is not None:
                    p_pct = float(p)

        route_points.append({
            "lat": float(lat),
            "lon": float(lon),
            "risk_pct": float(p_pct),
        })

    # 4) indeks tveganja poti = uteženo povprečje (po času)
    n = max(1, len(route_points) - 1)
    dt_min = (duration_s / 60.0) / n  # približne minute na segment

    weighted_sum = 0.0
    weight_total = 0.0

    for pt in route_points:
        p = float(pt.get("risk_pct") or 0.0)  # 0..100
        weighted_sum += p * dt_min
        weight_total += dt_min

    risk_index = (weighted_sum / weight_total) if weight_total else 0.0

    return jsonify({
        "status": "ok",
        "date": hourly_data.get("date"),
        "distance_km": round(distance_m / 1000.0, 2),
        "duration_min": round(duration_s / 60.0, 1),
        "risk_probability": round(risk_index, 2),
        "route": route_points,
        "note": "To je indeks tveganja poti (uteženo povprečje urnih napovedi ob poti), ne osebna verjetnost."
    })
if __name__ == "__main__":
    app.run(debug=True, port=5001)