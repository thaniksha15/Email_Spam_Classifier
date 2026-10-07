from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    message = data["message"]

    # Convert message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    return jsonify({
        "message": message,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)