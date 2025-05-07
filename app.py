from flask import Flask, request, jsonify
import joblib
import json
import os
from utils.train_model import MODEL_FILE_PATH, METRICS_FILE_PATH

# Load the trained model
model = joblib.load(MODEL_FILE_PATH)

# Load model metrics if available
model_metrics = {}
if os.path.exists(METRICS_FILE_PATH):
    with open(METRICS_FILE_PATH, "r") as f:
        model_metrics = json.load(f)

def generate_recommendation(sentiment: str) -> str:
    if sentiment == "negative":
        return "Check common issues like delivery, quality, or customer service."
    elif sentiment == "neutro":
        return "Consider improving the experience to stand out more."
    elif sentiment == "positive":
        return "Keep reinforcing what you're doing well."
    return ""

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text is required"}), 400

    sentiment = model.predict([text])[0]
    recommendation = generate_recommendation(sentiment)

    return jsonify({"sentiment": sentiment, "recommendation": recommendation})

@app.route("/metrics", methods=["GET"])
def metrics():
    if not model_metrics:
        return jsonify({"error": "Model metrics not found."}), 404
    return jsonify(model_metrics)

if __name__ == "__main__":
    app.run(debug=True)
