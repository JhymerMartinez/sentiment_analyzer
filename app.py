from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load("sentiment_model.pkl")

def generate_recommendation(sentiment: str) -> str:
    if sentiment == "negativo":
        return "Check common issues like delivery, quality, or customer service."
    elif sentiment == "neutro":
        return "Consider improving the experience to stand out more."
    elif sentiment == "positivo":
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

if __name__ == "__main__":
    app.run(debug=True)
