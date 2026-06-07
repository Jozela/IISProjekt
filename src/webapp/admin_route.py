import os
import json
from functools import wraps
from flask import Blueprint, request, session, redirect, render_template_string, send_file

admin_bp = Blueprint("admin", __name__)

ADMIN_USER = "admin"
ADMIN_PASS = "admin"

SHAP_PATH        = os.environ.get("SHAP_PATH",        "/app/models/shap_per_obcina.json")
PREDICTIONS_PATH = os.environ.get("PREDICTIONS_PATH", "/app/data/predictions/next24h.json")
REPORT_PATH      = "/app/reports/nesrece_data_testing_report.html"

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect("/admin/login")
        return f(*args, **kwargs)
    return decorated

@admin_bp.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = ""
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and request.form.get("password") == ADMIN_PASS:
            session["admin"] = True
            return redirect("/admin")
        error = "Napačno geslo"
    return render_template_string("""
<!DOCTYPE html><html><head><title>Admin Login</title>
<style>
body{background:#0a0c10;color:#eee;font-family:monospace;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
.box{background:#13161c;border:1px solid #222;padding:40px;border-radius:8px;width:300px}
h2{color:#e63946;margin:0 0 24px}
input{width:100%;padding:10px;margin:8px 0;background:#0a0c10;border:1px solid #333;color:#eee;border-radius:4px;box-sizing:border-box}
button{width:100%;padding:10px;background:#e63946;color:#fff;border:none;border-radius:4px;cursor:pointer;margin-top:8px}
.err{color:#e63946;font-size:12px}
</style></head><body>
<div class="box">
<h2>🔐 ADMIN</h2>
<form method="POST">
<input name="username" placeholder="Uporabniško ime" required>
<input name="password" type="password" placeholder="Geslo" required>
<button type="submit">Prijava</button>
</form>
<p class="err">{{error}}</p>
</div></body></html>
""", error=error)

@admin_bp.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin/login")

@admin_bp.route("/admin")
@admin_required
def admin():
    shap_data = load_json(SHAP_PATH)
    predictions = load_json(PREDICTIONS_PATH)

    n_obcin = len(predictions.get("predictions_by_obcina", {}))
    generated_at = predictions.get("generated_at", "N/A")

    global_imp = shap_data.get("global_importance", {})
    top_features = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:10] if global_imp else []

    report_exists = os.path.exists(REPORT_PATH)

    return render_template_string("""
<!DOCTYPE html><html><head><title>Admin — Nadzorna plošča</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0c10;color:#eee;font-family:monospace;padding:24px}
h1{color:#e63946;margin-bottom:24px;font-size:22px}
h2{color:#aaa;font-size:14px;text-transform:uppercase;letter-spacing:2px;margin:24px 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}
.card{background:#13161c;border:1px solid #222;border-radius:8px;padding:20px}
.card .val{font-size:28px;font-weight:bold;color:#e63946;margin:8px 0}
.card .lbl{font-size:11px;color:#666;text-transform:uppercase}
table{width:100%;border-collapse:collapse;background:#13161c;border-radius:8px;overflow:hidden}
th{background:#1a1d24;padding:10px 16px;text-align:left;font-size:11px;color:#666;text-transform:uppercase}
td{padding:10px 16px;border-top:1px solid #1a1d24;font-size:13px}
.bar-wrap{background:#1a1d24;border-radius:4px;height:8px;width:200px;display:inline-block;vertical-align:middle}
.bar{background:#e63946;height:8px;border-radius:4px}
.nav{display:flex;gap:16px;margin-bottom:24px;align-items:center}
.nav a{color:#666;text-decoration:none;font-size:12px}
.nav a:hover{color:#eee}
.badge{padding:3px 8px;border-radius:4px;font-size:11px;background:#1e3a2f;color:#4caf50}
iframe{width:100%;height:600px;border:1px solid #222;border-radius:8px;background:#fff}
</style></head>
<body>
<div class="nav">
  <h1>🛡 Admin — Nadzorna plošča</h1>
  <a href="/">← Nazaj na aplikacijo</a>
  <a href="/admin/logout">Odjava</a>
</div>

<h2>Pregled sistema</h2>
<div class="grid">
  <div class="card"><div class="lbl">Občin v napovedi</div><div class="val">{{n_obcin}}</div></div>
  <div class="card"><div class="lbl">Zadnja generacija</div><div class="val" style="font-size:14px;margin-top:12px">{{generated_at[:16] if generated_at != 'N/A' else 'N/A'}}</div></div>
  <div class="card"><div class="lbl">Model</div><div class="val" style="font-size:16px;margin-top:12px">XGBoost</div></div>
  <div class="card"><div class="lbl">Status</div><div class="val" style="font-size:16px;margin-top:12px"><span class="badge">✓ V produkciji</span></div></div>
</div>

<h2>SHAP — Pomembnost značilk (globalno)</h2>
{% if top_features %}
<table>
<tr><th>#</th><th>Značilka</th><th>Pomembnost</th><th>Vizualizacija</th></tr>
{% set max_val = top_features[0][1] %}
{% for name, val in top_features %}
<tr>
  <td>{{loop.index}}</td>
  <td>{{name}}</td>
  <td>{{'{:.6f}'.format(val)}}</td>
  <td><div class="bar-wrap"><div class="bar" style="width:{{(val/max_val*100)|int}}%"></div></div></td>
</tr>
{% endfor %}
</table>
{% else %}
<p style="color:#666">Ni SHAP podatkov.</p>
{% endif %}

<h2>Poročilo o kakovosti podatkov</h2>
{% if report_exists %}
<iframe src="/admin/report"></iframe>
{% else %}
<p style="color:#666">Poročilo ni na voljo. Zaženite pipeline.</p>
{% endif %}

</body></html>
""", n_obcin=n_obcin, generated_at=generated_at, top_features=top_features, report_exists=report_exists)

@admin_bp.route("/admin/report")
@admin_required
def admin_report():
    if os.path.exists(REPORT_PATH):
        return send_file(REPORT_PATH)
    return "Poročilo ni na voljo.", 404