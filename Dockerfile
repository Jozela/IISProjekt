# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy the webapp source
COPY src/webapp/app.py                 src/webapp/app.py
COPY src/webapp/chat_route.py          src/webapp/chat_route.py
COPY src/webapp/build_api_artifacts.py src/webapp/build_api_artifacts.py
COPY src/webapp/templates/             src/webapp/templates/

# Copy the models package (NesreceWeatherPreprocessor)
COPY src/models/ src/models/

# Make src importable as a package
RUN touch src/__init__.py

# Bake in static data (centroids, geojson, raw) and trained models —
# these don't change day-to-day
COPY data/obcine_centroids.json data/obcine_centroids.json
COPY data/obcine.geojson        data/obcine.geojson
COPY models/                    models/

# predictions/ is intentionally NOT copied here —
# it is volume-mounted at runtime so daily updates are picked up
RUN mkdir -p data/predictions

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5001 \
    CENTROIDS_PATH=data/obcine_centroids.json \
    PREDICTIONS_PATH=data/predictions/next24h.json \
    HOURLY_PATH=data/predictions/hourly.json \
    SHAP_PATH=models/shap_per_obcina.json \
    MODEL_PATH=models/model_nesrece.pkl \
    PIPELINE_PATH=models/pipeline_nesrece.pkl \
    MERGER_PATH=models/merger_nesrece.pkl \
    SHAP_NPZ=models/shap_values.npz

EXPOSE 5001

WORKDIR /app/src/webapp

CMD ["sh", "-c", "python src/models/predict.py && gunicorn --bind 0.0.0.0:5001 --workers 2 --timeout 120 app:app"]