from flask import Flask, render_template_string, request

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Early Burnout Check</title>
  <style>
    body{font-family:Arial, sans-serif; max-width:800px; margin:40px auto; line-height:1.4}
    .card{border:1px solid #ddd; border-radius:12px; padding:20px; box-shadow:0 1px 6px rgba(0,0,0,.06)}
    label{font-weight:600}
    .row{margin-bottom:14px}
    input[type=range]{width:100%}
    .score{font-size:22px; font-weight:700}
    .low{color:#2e7d32}.med{color:#f9a825}.high{color:#c62828}
    textarea{width:100%; height:110px}
    button{background:#111;color:#fff;border:none;padding:10px 16px;border-radius:8px;cursor:pointer}
  </style>
</head>
<body>
  <h1>Early Burnout Detection (Demo)</h1>
  <p>Enter a short description of how you feel and adjust the sliders. The app returns a simple risk score (0–100) and a label.</p>
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

      <button type="submit">Check Risk</button>
    </form>
  </div>

  {% if score is not none %}
  <div class="card" style="margin-top:18px">
    <div class="score {{ label_class }}">Risk Score: {{ score }} / 100 — {{ label }}</div>
    <p><strong>Explanation:</strong> {{ explanation }}</p>
  </div>
  {% endif %}
</body>
</html>
"""

NEG_KEYS = {
    "exhausted": 10, "tired": 8, "overwhelmed": 12, "anxious": 8, "stressed": 10,
    "burnout": 15, "deadline": 6, "pressure": 7, "sleep": 6, "insomnia": 10,
    "depressed": 14, "fatigue": 10, "cynical": 8, "drained": 10
}
POS_KEYS = {"supported": -8, "energized": -8, "motivated": -6, "rested": -8, "balanced": -10}

def rule_score(text, hours, workload, sleep, stress):
    text = (text or "").lower()
    base = 0
    matched = []

    for k, v in NEG_KEYS.items():
        if k in text:
            base += v; matched.append(k)
    for k, v in POS_KEYS.items():
        if k in text:
            base += v; matched.append(k)

    base += max(0, (hours - 45)) * 0.9
    base += (workload - 5) * 2.5
    base += (stress - 5) * 3.0
    base += max(0, (6 - sleep)) * 3.0

    score = int(max(0, min(100, round(base))))
    return score, matched

def label_from_score(s):
    if s < 35: return "Low", "low"
    if s < 70: return "Moderate", "med"
    return "High", "high"

@app.route("/", methods=["GET","POST"])
def home():
    ctx = {"score": None}
    if request.method == "POST":
        free_text = request.form.get("free_text", "")
        hours = int(request.form.get("hours", 45))
        workload = int(request.form.get("workload", 6))
        sleep = int(request.form.get("sleep", 6))
        stress = int(request.form.get("stress", 5))

        score, matched = rule_score(free_text, hours, workload, sleep, stress)
        label, css = label_from_score(score)

        why = []
        if matched: why.append(f"Text signals detected: {', '.join(matched)}.")
        if hours > 50: why.append("Extended weekly hours.")
        if workload >= 7: why.append("High perceived workload.")
        if stress >= 7: why.append("High stress.")
        if sleep <= 4: why.append("Poor sleep quality.")
        if not why: why.append("Inputs within typical ranges.")

        ctx.update(dict(
            score=score, label=label, label_class=css, explanation=" ".join(why),
            free_text=free_text, hours=hours, workload=workload, sleep=sleep, stress=stress
        ))
    return render_template_string(HTML, **ctx)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
