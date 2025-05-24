import re
import joblib

# Load model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    return text

# Clean examples only — no sarcasm, irony, or ambiguity
test_texts = [
    "Just got a promotion at work. Feeling really happy!",
    "Lost my job today and I feel devastated.",
    "My dog passed away last night. I'm heartbroken.",
    "Had a great lunch with my friends. So cheerful!",
    "Woke up to the sound of birds. Feeling peaceful and glad.",
    "It’s been a rough day and I can’t stop crying.",
]

# Predict emotion
for text in test_texts:
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    print(f"Text: {text}\n→ Detected Emotion: {pred}\n")
