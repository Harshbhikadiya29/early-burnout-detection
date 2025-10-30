# app/app.py
from flask import Flask, render_template_string, request
import os
from project.predict import predict as predict_fn

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Early Burnout Prediction</title>
  <style>
    body{font-family:Arial, sans-serif; max-width:820px; margin:36px auto}
    .card{border:1px solid #e5e7eb; border-radius:14px; padding:20px; box-shadow:0 1px 6px rgba(0,0,0,.06)}
    .row{margin-bottom:14px}
    label{font-weight:600}
    textarea{width:100%; height:110px}
    input[type=range]{width:100%}
    button{background:#111;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer}
    .score{font-size:22px; font-weight:700}
    .low{color:#2e7d32}.med{color:#f9a825}.high{color:#c62828}
    small{color:#6b7280}
  </style>
</head>
<body>
  <h1>Early Burnout Prediction</h1>
  <p><small>Model path: <code>{{ model_path }}</code>. If unavailable, a safe fallback rules engine is used.</small></p>

  <div class="card">
    <form method="post" action="/">
      <div class="row">
        <label for="free_text">How are you feeling lately?</label><br/>
        <textarea id="free_text" name="free_text" placeholder="e.g., I'm exhausted, sleeping poorly, deadlines piling up...">{{ free_text or "" }}</textarea>
      </div>

      <div class="row">
        <label>Weekly hours (30–80):</label>
        <input type="range" min="30" max="80" value="{{ hours or 45 }}" name="hours" oninput="h.value=this.value"><output id="h">{{ hours or 45 }}</output>
      </div>

      <div class="row">
        <label>Perceived workload (1–10):</label>
        <input type="range" min="1" max="10" value="{{ workload or 6 }}" name="workload" oninput="w.value=this.value"><output id="w">{{ workload or 6 }}</output>
      </div>

      <div class="row">
        <label>Sleep quality (1=poor, 10=great):</label>
        <input type="range" min="1" max="10" value="{{ sleep or 6 }}" name="sleep" oninput="s.value=this.value"><output id="s">{{ sleep or 6 }}</output>
      </div>

      <div class="row">
        <label>Stress level (1–10):</label>
        <input type="range" min="1" max="10" value="{{ stress or 5 }}" name="stress" oninput="t.value=this.value"><output id="t">{{ stress or 5 }}</output>
      </div>

      <button type="submit">Predict</button>
    </form>
  </div>

  {% if result %}
  <div class="card" style="margin-top:18px">
    {% set css = 'low' if result.label=='Low' else ('med' if result.label=='Moderate' else 'high') %}
    <div class="score {{ css }}">Risk Score: {{ result.score }} / 100 — {{ result.label }}</div>
    <p>
      {% if result.proba is not none %}
        Model probability: {{ '%.2f'|format(result.proba) }}
      {% else %}
        <em>No probability (fallback or model without predict_proba).</em>
      {% endif %}
      {% if result.used_fallback %}
        <br><small>Fallback rules used (trained model missing or incompatible).</small>
      {% endif %}
    </p>
  </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    ctx = {
        "result": None,
        "free_text": "",
        "hours": 45, "workload": 6, "sleep": 6, "stress": 5,
        "model_path": os.getenv("MODEL_PATH", "model/burnout_model.pkl"),
    }
    if request.method == "POST":
        payload = {
            "free_text": request.form.get("free_text", ""),
            "hours": request.form.get("hours", 45),
            "workload": request.form.get("workload", 6),
            "sleep": request.form.get("sleep", 6),
            "stress": request.form.get("stress", 5),
        }
        res = predict_fn(payload)
        class R: pass
        r = R(); r.score=res["score"]; r.label=res["label"]; r.proba=res["proba"]; r.used_fallback=res["used_fallback"]
        ctx.update(payload); ctx["result"] = r
    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
