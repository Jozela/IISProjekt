import json

INP = "data/obcine.geojson"   # shrani geojson lokalno (ne fetch vsakokrat)
OUT = "data/obcine_centroids.json"

def centroid_of_polygon(coords):
    # coords: list of [lon,lat] points (rough centroid approx for polygon ring)
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (sum(ys)/len(ys), sum(xs)/len(xs))  # lat, lon

with open(INP, encoding="utf-8") as f:
    gj = json.load(f)

centroids = {}
for feat in gj["features"]:
    props = feat["properties"]
    name = props.get("OB_UIME") or props.get("name")
    geom = feat["geometry"]
    if not name or not geom:
        continue

    if geom["type"] == "Polygon":
        ring = geom["coordinates"][0]
        lat, lon = centroid_of_polygon(ring)
    elif geom["type"] == "MultiPolygon":
        ring = geom["coordinates"][0][0]
        lat, lon = centroid_of_polygon(ring)
    else:
        continue

    centroids[name] = [lat, lon]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(centroids, f, ensure_ascii=False, indent=2)

print("Wrote", OUT, "count:", len(centroids))