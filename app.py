
from flask import Flask, render_template, request
import pickle
import json, os
from collections import Counter
from datetime import datetime

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Emoji map
emoji_map = {
    "joy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "suprise": "😲"
}

# File to store emotion history
HISTORY_FILE = "emotion_history.json"

# Save prediction to history file
def save_prediction_to_history(emotion):
    record = {
        "emotion": emotion,
        "timestamp": datetime.now().isoformat()
    }
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
    with open(HISTORY_FILE, "r+") as f:
        data = json.load(f)
        data.append(record)
        f.seek(0)
        json.dump(data, f)

# Count emotions
def get_emotion_trend():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
        emotions = [entry["emotion"] for entry in history]
        return dict(Counter(emotions))

# Main route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    emoji = ""
    text = ""
    confidence = None
    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = round(max(proba) * 100, 2)
            emoji = emoji_map.get(prediction.lower(), "🤔")
            save_prediction_to_history(prediction.lower())
    trend_data = get_emotion_trend()
    return render_template("index.html", prediction=prediction, emoji=emoji, text=text, confidence=confidence, trend=trend_data)

if __name__ == "__main__":
    app.run(debug=True)

