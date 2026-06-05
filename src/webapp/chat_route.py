"""
Dodaj ta blueprint v app.py:

    from chat_route import chat_bp
    app.register_blueprint(chat_bp)

Potrebuješ: pip install google-genai
Nastavi env spremenljivko: GEMINI_API_KEY=...
"""

import os
import json
import requests
from flask import Blueprint, request, jsonify

chat_bp = Blueprint("chat", __name__)

SYSTEM_PROMPT = """Si prometni varnostni asistent za Slovenijo. Tvoje ime je "VariPot".

Odgovarjaš SAMO na vprašanja, ki so neposredno povezana z:
- Prometno varnostjo v Sloveniji
- Verjetnostjo nesreč po občinah in cestah
- Vremenskimi razmerami in vplivom na promet
- Priporočili za vožnjo (kdaj je varno/nevarno)
- Alternativnimi potmi in izogibanjem nevarnim odsekom
- Razlago podatkov o tveganjih (kaj pomeni 70% tveganje itd.)

Na vprašanja, ki niso vezana na te teme, prijazno odgovori:
"Oprosti, lahko ti pomagam samo pri vprašanjih o prometni varnosti in poteh v Sloveniji."

Pravila:
- Vedno odgovarjaj v slovenščini
- Bodi jedrnat in jasen (max 3-4 stavki za preproste odgovore)
- Kadar omeniš tveganje, razloži kaj to pomeni praktično
- Visoko tveganje (>50%) = odsvetuj pot ali priporoči previdnost
- Zmerno tveganje (20-50%) = vozi previdno
- Nizko tveganje (<20%) = normalna vožnja

Podatki o tveganjih so izračunani iz ML modela, ki upošteva zgodovinske nesreče,
vremenske razmere in urne vzorce prometa. Niso absolutna resnica — so statistična napoved.
"""

# Poti do JSON datotek — enako kot v app.py
PREDICTIONS_PATH = os.environ.get("PREDICTIONS_PATH", "data/predictions/today.json")
HOURLY_PATH      = os.environ.get("HOURLY_PATH",      "data/predictions/hourly.json")

def load_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def avg_hours(hourly_list, hrs):
    vals = [hourly_list[h] for h in hrs if h < len(hourly_list) and hourly_list[h] is not None]
    return sum(vals) / len(vals) if vals else None

def peak_hours(hourly_list, n=3):
    indexed = [(h, p) for h, p in enumerate(hourly_list) if p is not None]
    return sorted(indexed, key=lambda x: x[1], reverse=True)[:n]

def build_data_context(context: dict) -> str:
    parts = []

    # 1) Vse obcine z dnevnim tveganjem
    today_data = load_json_file(PREDICTIONS_PATH)
    preds = today_data.get("predictions", {})

    if preds:
        sorted_preds = sorted(
            [(k, v) for k, v in preds.items() if v is not None],
            key=lambda x: x[1], reverse=True
        )
        all_str = "\n".join(f"  {n}: {v:.1f}%" for n, v in sorted_preds)
        parts.append(f"DNEVNO TVEGANJE VSEH OBCIN (ML napoved, razvrsceno po tveganju):\n{all_str}")

    # 2) Urne napovedi za VSE obcine direktno iz hourly.json
    hourly_data = load_json_file(HOURLY_PATH)
    hourly_preds = hourly_data.get("predictions", {})

    if hourly_preds:
        lines = []
        ordered = sorted(
            hourly_preds.items(),
            key=lambda x: preds.get(x[0], 0) or 0,
            reverse=True
        )
        for obcina, hourly in ordered:
            if not hourly:
                continue
            pk = peak_hours(hourly, 3)
            if not pk:
                continue
            peak_str = ", ".join(f"{h:02d}:00={p:.0f}%" for h, p in pk)
            a_noc   = avg_hours(hourly, range(0, 6))
            a_jutro = avg_hours(hourly, range(6, 10))
            a_dan   = avg_hours(hourly, range(10, 16))
            a_vecer = avg_hours(hourly, range(16, 22))
            day_parts = []
            if a_noc   is not None: day_parts.append(f"noc={a_noc:.0f}%")
            if a_jutro is not None: day_parts.append(f"jutro={a_jutro:.0f}%")
            if a_dan   is not None: day_parts.append(f"dan={a_dan:.0f}%")
            if a_vecer is not None: day_parts.append(f"vecer={a_vecer:.0f}%")
            lines.append(f"  {obcina}: peak({peak_str}) | {', '.join(day_parts)}")

        if lines:
            parts.append("URNE NAPOVEDI VSEH OBCIN:\n" + "\n".join(lines))

    # 3) Izbrana obcina iz frontenda
    selected = context.get("selected_obcina")
    if selected:
        prob = context.get("prob")
        hourly = hourly_preds.get(selected) or hourly_preds.get(
            next((k for k in hourly_preds if k.lower() == selected.lower()), None), None
        )
        detail = f"Trenutno prikazana obcina: {selected}"
        if prob is not None:
            detail += f" — dnevno tveganje {prob:.1f}%"
        if hourly:
            pk = peak_hours(hourly, 3)
            detail += " — najnevarnejse ure: " + ", ".join(f"{h:02d}:00 ({p:.1f}%)" for h, p in pk)
        parts.append(detail)

    # 4) Pot
    if context.get("route_risk") is not None:
        parts.append(
            f"Zadnja izracunana pot: tveganje {context['route_risk']}% | "
            f"{context.get('distance_km', '?')} km | "
            f"{context.get('duration_min', '?')} min"
            + (f" | cilj: {context['dest']}" if context.get('dest') else "")
        )

    if not parts:
        return ""
    return "\n\nAKTUALNI PODATKI IZ ML MODELA (uporabi jih pri odgovorih):\n" + "\n\n".join(parts)


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return jsonify({"error": "Namesti: pip install google-genai"}), 500

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY ni nastavljen"}), 500

    body     = request.get_json(force=True) or {}
    messages = body.get("messages", [])
    context  = body.get("context", {})

    if not messages:
        return jsonify({"error": "Manjkajo sporocila"}), 400

    system = SYSTEM_PROMPT + build_data_context(context)

    client = genai.Client(api_key=api_key)

    history = []
    for m in messages[:-1]:
        role    = m.get("role")
        content = m.get("content", "")
        if role == "user":
            history.append(types.Content(role="user",  parts=[types.Part(text=content)]))
        elif role == "assistant":
            history.append(types.Content(role="model", parts=[types.Part(text=content)]))

    last_msg = messages[-1].get("content", "")

    chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
        history=history,
    )

    response = chat_session.send_message(last_msg)
    reply = response.text or "Napaka pri odgovoru."
    return jsonify({"reply": reply})