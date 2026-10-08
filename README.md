# 📧 Email Spam Classifier

A Machine Learning based web application that classifies email or text messages as **SPAM** or **HAM (Not Spam)**.

## 🚀 Live Demo

👉 [Click here to try the Email Spam Classifier](https://email-spam-classifier-c0tn.onrender.com)

## 📌 Project Overview

Email Spam Classifier uses **Machine Learning** and **Natural Language Processing (NLP)** techniques to identify unwanted or spam messages.

The system converts the input message into numerical features using **TF-IDF Vectorization** and uses a trained Machine Learning model to classify the message as **SPAM** or **HAM**.

## ✨ Features

* 📩 Classifies messages as **SPAM** or **HAM**
* 🤖 Machine Learning based prediction
* 🔤 TF-IDF text vectorization
* 🌐 Flask web application
* 💻 Simple and user-friendly interface
* 📊 Model accuracy of approximately **97.49%**
* 🚀 Deployed using **Render**

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **TF-IDF**
* **Flask**
* **Joblib**
* **HTML**
* **CSS**
* **JavaScript**
* **Render**

## 📂 Dataset

The project uses the **SMS Spam Collection Dataset**.

### Dataset Details

* **Total Messages:** 5572
* **Ham Messages:** 4825
* **Spam Messages:** 747

## 🧠 Machine Learning Model

The text messages are converted into numerical features using **TF-IDF Vectorization**.

The trained Machine Learning model then analyzes these features and classifies new messages into:

* **HAM** – Normal or legitimate message
* **SPAM** – Unwanted or suspicious message

## 📊 Model Performance

The trained model achieved approximately:

**Accuracy: 97.49%**

This shows that the model can effectively distinguish between spam and legitimate messages.

## 📁 Project Structure

```text
Email_Spam_Classifier/
│
├── dataset/
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
```

## ⚙️ How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/thaniksha15/Email_Spam_Classifier.git
```

### 2. Navigate to the Project Folder

```bash
cd Email_Spam_Classifier
```

### 3. Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

### 5. Open in Browser

```text
http://127.0.0.1:5000
```

## 🧪
