
from flask import Flask, render_template, request
import pickle
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Emoji mapping for each emotion label
emoji_map = {
    "joy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "suprise": "😲"
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    emoji = ""
    text = ""
    confidence = None

    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            # Vectorize and predict
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]
            # Get confidence score
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vec)[0]
                confidence = round(max(probas) * 100, 2)
            else:
                confidence = None
            # Get emoji for predicted emotion
            emoji = emoji_map.get(prediction.lower(), "🤔")

    return render_template("index.html", 
                           prediction=prediction, 
                           emoji=emoji, 
                           confidence=confidence, 
                           text=text)

if __name__ == "__main__":
    app.run(debug=True)

