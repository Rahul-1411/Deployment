
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Emotion to emoji map (only your 6 labels)
emoji_map = {
    "joy": "😊",
    "sad": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "suprise": "😲"  # Keep same spelling if your model used it
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    emoji = ""
    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]
            emoji = emoji_map.get(prediction.lower(), "🤔")
    return render_template("index.html", prediction=prediction, emoji=emoji)

if __name__ == "__main__":
    app.run(debug=True)
