import re
import joblib


MODEL_PATH = "models/spam_pipeline.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


model = joblib.load(MODEL_PATH)

print("=" * 60)
print("REAL-TIME SMS SPAM DETECTOR")
print("=" * 60)

while True:

    message = input("\nEnter SMS message: ")

    if message.lower() in ["exit", "quit"]:
        print("Program stopped.")
        break

    cleaned_message = clean_text(message)

    prediction = model.predict([cleaned_message])[0]

    probabilities = model.predict_proba([cleaned_message])[0]

    ham_probability = probabilities[0]
    spam_probability = probabilities[1]

    if prediction == 1:
        result = "SPAM"
        confidence = spam_probability
    else:
        result = "HAM"
        confidence = ham_probability

    print("\nPrediction :", result)
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Ham        : {ham_probability * 100:.2f}%")
    print(f"Spam       : {spam_probability * 100:.2f}%")