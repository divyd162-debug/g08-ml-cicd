import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "models" / "ticket_classifier.pkl"
)

tfidf = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)


subject = "Cannot login to my account"

body = (
    "I have entered the correct email and password "
    "but I am unable to log into my account"
)

text = subject + " " + body

X = tfidf.transform([text])


prediction = model.predict(X)[0]

probabilities = model.predict_proba(X)[0]

classes = model.classes_


print("\nPrediction:")
print(prediction)

print("\nProbabilities:")

for department, probability in sorted(
    zip(classes, probabilities),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{department}: "
        f"{probability * 100:.2f}%"
    )