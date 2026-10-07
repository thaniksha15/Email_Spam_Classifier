import joblib

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Email Spam Classifier")
print("---------------------")

message = input("Enter your message: ")

# Convert message into TF-IDF features
message_tfidf = vectorizer.transform([message])

# Make prediction
prediction = model.predict(message_tfidf)[0]

if prediction == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")