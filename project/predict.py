# project/predict.py
import os
import pathlib
import joblib
import numpy as np

# Where to load the trained model from
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "model/burnout_model.pkl")
_model = None

# --------- Fallback rules (used only if model can't load) ----------
NEG = ["exhausted","tired","overwhelmed","anxious","stressed","burnout",
       "deadline","pressure","insomnia","depressed","fatigue","cynical","drained","sleep"]
POS = ["supported","energized","motivated","rested","balanced"]

def _fallback(payload):
    text = (payload.get("free_text") or "").lower()
    hours = float(payload.get("hours", 45))
    workload = float(payload.get("workload", 6))
    sleep = float(payload.get("sleep", 6))
    stress = float(payload.get("stress", 5))

    score = 0
    score += sum(10 for k in NEG if k in text)
    score -= sum(8  for k in POS if k in text)
    score += max(0, (hours - 45)) * 0.9
    score += (workload - 5) * 2.5
    score += (stress   - 5) * 3.0
    score += max(0, (6 - sleep)) * 3.0
    score = int(max(0, min(100, round(score))))
    label = "Low" if score < 35 else ("Moderate" if score < 70 else "High")
    return {"score": score, "label": label, "proba": None, "used_fallback": True}

# --------- Model loading ----------
def _load_model(path=DEFAULT_MODEL_PATH):
    global _model
    if _model is not None:
        return _model
    p = pathlib.Path(path)
    if not p.exists():
        return None
    _model = joblib.load(p.as_posix())
    return _model

def _prob_of_positive(model, probs):
    """
    Return probability of the positive class (class '1' if available).
    Avoid np.max(...) which makes everything look ~0.99.
    """
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        idx = classes.index(1)
    elif True in classes:
        idx = classes.index(True)
    else:
        # Fallback to the last column being "positive"
        idx = len(classes) - 1
    return float(probs[idx])

# --------- Public API ----------
def predict(payload: dict, model_path: str = DEFAULT_MODEL_PATH):
    """
    Expects keys: free_text, hours, workload, sleep, stress
    Tries numeric model first (hours, workload, sleep, stress).
    If that fails, tries a text combo.
    Else falls back to rules.
    """
    model = _load_model(model_path)
    if model is None:
        return _fallback(payload)

    # Numeric pathway (matches train_model.py)
    try:
        X = np.array([[
            float(payload.get("hours", 45)),
            float(payload.get("workload", 6)),
            float(payload.get("sleep", 6)),
            float(payload.get("stress", 5)),
        ]], dtype=float)

        yhat = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = _prob_of_positive(model, model.predict_proba(X)[0])
        # Map probability of positive class to 0–100
        if proba is not None:
            score = int(round(proba * 100))
        else:
            score = 100 if int(yhat) == 1 else 0

        label = "High" if score >= 70 else ("Moderate" if score >= 35 else "Low")
        return {"score": score, "label": label, "proba": proba, "used_fallback": False}
    except Exception:
        pass

    # Text pathway (if your model is a text pipeline)
    try:
        text = str(payload.get("free_text", "")).strip()
        combo = (
            f"{text} hours={payload.get('hours', 45)} "
            f"workload={payload.get('workload', 6)} "
            f"sleep={payload.get('sleep', 6)} "
            f"stress={payload.get('stress', 5)}"
        )
        yhat = model.predict([combo])[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = _prob_of_positive(model, model.predict_proba([combo])[0])
        if proba is not None:
            score = int(round(proba * 100))
        else:
            score = 100 if int(yhat) == 1 else 0

        label = "High" if score >= 70 else ("Moderate" if score >= 35 else "Low")
        return {"score": score, "label": label, "proba": proba, "used_fallback": False}
    except Exception:
        # Last resort
        return _fallback(payload)
