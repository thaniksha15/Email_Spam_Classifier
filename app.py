from flask import Flask, request, jsonify, render_template
import joblib
from collections import Counter
from datetime import datetime

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

prediction_history = []
stats = Counter()

SPAM_KEYWORDS = [
    "win", "winner", "won", "prize", "free",
    "claim", "urgent", "offer", "cash", "reward",
    "click", "congratulations", "selected",
    "limited time", "call now"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Please enter a message."}), 400

    message = message.strip()

    try:
        message_tfidf = vectorizer.transform([message])
        prediction = int(model.predict(message_tfidf)[0])
        result = "SPAM" if prediction == 1 else "HAM"

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(message_tfidf)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        lower_message = message.lower()
        matched_keywords = [
            word for word in SPAM_KEYWORDS
            if word in lower_message
        ]

        record = {
            "message": message,
            "prediction": result,
            "confidence": confidence,
            "keywords": matched_keywords,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        prediction_history.insert(0, record)
        del prediction_history[100:]
        stats[result] += 1

        return jsonify(record)

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please try another message."
        }), 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify(prediction_history)


@app.route("/stats", methods=["GET"])
def statistics():
    total = stats["SPAM"] + stats["HAM"]

    return jsonify({
        "total": total,
        "spam": stats["SPAM"],
        "ham": stats["HAM"]
    })


@app.route("/clear-history", methods=["POST"])
def clear_history():
    prediction_history.clear()
    stats.clear()

    return jsonify({
        "message": "History and statistics cleared."
    })


if __name__ == "__main__":
    app.run(debug=True)
