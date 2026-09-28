import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE = Path(__file__).resolve().parent

model = joblib.load(BASE / "final_model.pkl")
scaler = joblib.load(BASE / "scaler.pkl")
features = json.load(open(BASE / "feature_names.json"))
threshold = json.load(open(BASE / "threshold.json"))["threshold"]

app = Flask(__name__)

def risk_band(prob: float) -> str:
    if prob < threshold:
        return "Low"
    elif prob < 0.5:
        return "Medium"
    return "High"

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    row = {f: payload.get(f, 0) for f in features}
    input_df = pd.DataFrame([row], columns=features)
    scaled = scaler.transform(input_df)
    prob = float(model.predict_proba(scaled)[0, 1])
    pred = int(prob >= threshold)
    return jsonify({
        "probability": prob,
        "predicted_binary": pred,
        "risk_level": risk_band(prob),
        "threshold": threshold,
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)