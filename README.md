# Email Spam Classifier

A Machine Learning based web application that classifies email or text messages as **SPAM** or **HAM (Not Spam)**.

## Project Overview

Email Spam Classifier uses Machine Learning and Natural Language Processing (NLP) techniques to identify unwanted or spam messages.

The system converts the input message into numerical features using **TF-IDF Vectorization** and uses a trained Machine Learning model to classify the message.

## Features

- Classifies messages as SPAM or HAM
- Machine Learning based prediction
- TF-IDF text vectorization
- Flask web application
- Simple and user-friendly interface
- Model accuracy of approximately 97.49%

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Flask
- Joblib
- HTML
- CSS
- JavaScript

## Dataset

The project uses the **SMS Spam Collection dataset**.

Dataset contains:

- Total messages: 5572
- Ham messages: 4825
- Spam messages: 747

## Machine Learning Model

The text messages are converted into numerical features using **TF-IDF Vectorization**.

The trained model is then used to classify new messages into:

- **HAM** – Normal message
- **SPAM** – Unwanted or suspicious message

## Model Performance

The trained model achieved approximately:

**Accuracy: 97.49%**

## Project Structure

```text
Email_Spam_Classifier/
│
├── templates/
│   └── index.html
│
├── app.py
├── predict.py
├── test_dataset.py
├── train_model.py
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── Procfile
└── README.md

## 🚀 Live Demo

[Click here to try the Email Spam Classifier](https://email-spam-classifier-c0tn.onrender.com)
