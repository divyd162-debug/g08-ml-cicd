import joblib
from pathlib import Path
from scipy.sparse import hstack


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def test_ticket_prediction():

    model = joblib.load(
        MODEL_DIR / "ticket_classifier_best_svm.pkl"
    )

    word_tfidf = joblib.load(
        MODEL_DIR / "tfidf_vectorizer_best_svm.pkl"
    )

    char_tfidf = joblib.load(
        MODEL_DIR / "char_tfidf_vectorizer_best_svm.pkl"
    )

    subject = "Cannot login to my account"

    body = (
        "I have entered the correct email and password "
        "but I am unable to log into my account"
    )

    text = subject + " " + body

    X_word = word_tfidf.transform([text])
    X_char = char_tfidf.transform([text])

    X = hstack([X_word, X_char])

    prediction = model.predict(X)[0]

    print("\nPrediction:", prediction)

    assert prediction in model.classes_

    print("Test passed: ticket prediction is working.")


if __name__ == "__main__":
    test_ticket_prediction()
