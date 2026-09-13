from flask import Flask, render_template, request
import joblib
import re
from pathlib import Path
import importlib.util

import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# G-08
# Automated Customer Support Ticket Routing System
# Flask Web Application
#
# FINAL PREDICTION MODEL:
# LinearSVC
# C = 3
# Word TF-IDF + Character TF-IDF
#
# PROBABILITY MODEL:
# Calibrated LinearSVC
# C = 3
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"


# ============================================================
# ORIGINAL BEST C=3 MODEL
# ============================================================

WORD_TFIDF_FILE = (
    MODEL_DIR / "tfidf_vectorizer_best_svm.pkl"
)

CHAR_TFIDF_FILE = (
    MODEL_DIR / "char_tfidf_vectorizer_best_svm.pkl"
)

MODEL_FILE = (
    MODEL_DIR / "ticket_classifier_best_svm.pkl"
)


# ============================================================
# CALIBRATED MODEL
# ============================================================

CALIBRATED_WORD_TFIDF_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer_calibrated.pkl"
)

CALIBRATED_CHAR_TFIDF_FILE = (
    MODEL_DIR
    / "char_tfidf_vectorizer_calibrated.pkl"
)

CALIBRATED_MODEL_FILE = (
    MODEL_DIR
    / "ticket_classifier_calibrated.pkl"
)


# ============================================================
# RESULT FILES
# ============================================================

ML_RESULTS_FILE = (
    RESULTS_DIR / "best_svm_predictions.csv"
)

BASELINE_RESULTS_FILE = (
    RESULTS_DIR / "heuristic_predictions.csv"
)

MODEL_COMPARISON_FILE = (
    RESULTS_DIR / "model_comparison.csv"
)


# ============================================================
# HEURISTIC BASELINE
# ============================================================

HEURISTIC_FILE = (
    SRC_DIR / "heuristic_baseline.py"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [

    WORD_TFIDF_FILE,

    CHAR_TFIDF_FILE,

    MODEL_FILE,

    CALIBRATED_WORD_TFIDF_FILE,

    CALIBRATED_CHAR_TFIDF_FILE,

    CALIBRATED_MODEL_FILE,

    HEURISTIC_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n\n"
            "Run src\\train_calibrated.py first."
        )


# ============================================================
# LOAD HEURISTIC BASELINE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "heuristic_baseline",
    HEURISTIC_FILE
)

heuristic_module = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    heuristic_module
)

predict_queue = (
    heuristic_module.predict_queue
)


# ============================================================
# LOAD ORIGINAL C=3 MODEL
# ============================================================

print("=" * 70)

print(
    "G-08 AUTOMATED CUSTOMER SUPPORT "
    "TICKET ROUTING SYSTEM"
)

print("=" * 70)

print("\nLoading FINAL C=3 prediction model...")

print("Model: LinearSVC")
print("C: 3")
print("Features: Word TF-IDF + Character TF-IDF")


word_tfidf = joblib.load(
    WORD_TFIDF_FILE
)

char_tfidf = joblib.load(
    CHAR_TFIDF_FILE
)

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# LOAD CALIBRATED MODEL
# ============================================================

print("\nLoading calibrated C=3 probability model...")

calibrated_word_tfidf = joblib.load(
    CALIBRATED_WORD_TFIDF_FILE
)

calibrated_char_tfidf = joblib.load(
    CALIBRATED_CHAR_TFIDF_FILE
)

calibrated_model = joblib.load(
    CALIBRATED_MODEL_FILE
)


print("\nAll models loaded successfully.")

print(
    f"Original Word features     : "
    f"{len(word_tfidf.vocabulary_)}"
)

print(
    f"Original Character features: "
    f"{len(char_tfidf.vocabulary_)}"
)

print(
    f"Calibrated Word features     : "
    f"{len(calibrated_word_tfidf.vocabulary_)}"
)

print(
    f"Calibrated Character features: "
    f"{len(calibrated_char_tfidf.vocabulary_)}"
)

print("=" * 70)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CREATE ORIGINAL MODEL FEATURES
# ============================================================

def transform_ticket(text):

    word_features = (
        word_tfidf.transform(
            [text]
        )
    )

    char_features = (
        char_tfidf.transform(
            [text]
        )
    )

    combined_features = hstack(
        [
            word_features,
            char_features
        ]
    ).tocsr()

    return combined_features


# ============================================================
# CREATE CALIBRATED MODEL FEATURES
# ============================================================

def transform_calibrated_ticket(text):

    word_features = (
        calibrated_word_tfidf.transform(
            [text]
        )
    )

    char_features = (
        calibrated_char_tfidf.transform(
            [text]
        )
    )

    combined_features = hstack(
        [
            word_features,
            char_features
        ]
    ).tocsr()

    return combined_features


# ============================================================
# CALCULATE REAL PROBABILITY ESTIMATES
# ============================================================

def calculate_probabilities(features):

    probability_values = (
        calibrated_model.predict_proba(
            features
        )[0]
    )

    classes = (
        calibrated_model.classes_
    )

    probability_data = []

    for department, probability in zip(
        classes,
        probability_values
    ):

        probability_data.append({

            "department":
                department,

            "probability":
                round(
                    float(
                        probability
                    ) * 100,
                    2
                )
        })


    # Highest probability first

    probability_data.sort(

        key=lambda x:
        x["probability"],

        reverse=True
    )


    return probability_data


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    return {

        "accuracy": round(
            accuracy_score(
                actual,
                predicted
            ) * 100,
            2
        ),

        "precision": round(
            precision_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        ),

        "recall": round(
            recall_score(
                actual,
                predicted,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        ),

        "f1": round(
            f1_score(
                actual,
                predicted,
                average="macro",
                zero_division=0
            ) * 100,
            2
        )
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    baseline_prediction = None

    probabilities = []

    agreement = None

    subject = ""

    body = ""

    error = None

    confidence = None


    # ========================================================
    # HANDLE POST REQUEST
    # ========================================================

    if request.method == "POST":

        subject = request.form.get(
            "subject",
            ""
        )

        body = request.form.get(
            "body",
            ""
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if (
            not subject.strip()
            and not body.strip()
        ):

            error = (
                "Please enter a ticket subject "
                "or ticket description."
            )

        else:

            # =================================================
            # COMBINE SUBJECT + BODY
            # =================================================

            ticket_text = (
                subject.strip()
                + " "
                + body.strip()
            )


            # =================================================
            # CLEAN TEXT
            # =================================================

            cleaned_text = clean_text(
                ticket_text
            )


            # =================================================
            # ORIGINAL MODEL FEATURES
            # =================================================

            combined_features = (
                transform_ticket(
                    cleaned_text
                )
            )


            # =================================================
            # FINAL C=3 ML PREDICTION
            # =================================================

            prediction = (
                model.predict(
                    combined_features
                )[0]
            )


            # =================================================
            # CALIBRATED MODEL FEATURES
            # =================================================

            calibrated_features = (
                transform_calibrated_ticket(
                    cleaned_text
                )
            )


            # =================================================
            # REAL PROBABILITY ESTIMATES
            # =================================================

            probabilities = (
                calculate_probabilities(
                    calibrated_features
                )
            )


            # =================================================
            # TOP PROBABILITY
            # =================================================

            if probabilities:

                confidence = (
                    probabilities[0][
                        "probability"
                    ]
                )


            # =================================================
            # HEURISTIC BASELINE
            # =================================================

            baseline_prediction = (
                predict_queue(
                    cleaned_text
                )
            )


            # =================================================
            # AGREEMENT
            # =================================================

            agreement = (
                prediction
                == baseline_prediction
            )


    # ========================================================
    # RENDER
    # ========================================================

    return render_template(

        "index.html",

        prediction=prediction,

        baseline_prediction=(
            baseline_prediction
        ),

        probabilities=probabilities,

        agreement=agreement,

        confidence=confidence,

        subject=subject,

        body=body,

        error=error
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route(
    "/dashboard"
)
def dashboard():

    # ========================================================
    # CHECK ML RESULT FILE
    # ========================================================

    if not ML_RESULTS_FILE.exists():

        return render_template(

            "dashboard.html",

            error=(
                "ML results file not found. "
                "Run tune_svm_c.py first."
            )
        )


    # ========================================================
    # CHECK BASELINE RESULT FILE
    # ========================================================

    if not BASELINE_RESULTS_FILE.exists():

        return render_template(

            "dashboard.html",

            error=(
                "Baseline results file not found. "
                "Run heuristic_baseline.py first."
            )
        )


    # ========================================================
    # LOAD RESULTS
    # ========================================================

    ml_df = pd.read_csv(
        ML_RESULTS_FILE
    )

    baseline_df = pd.read_csv(
        BASELINE_RESULTS_FILE
    )


    # ========================================================
    # ML METRICS
    # ========================================================

    ml_metrics = calculate_metrics(

        ml_df["actual_queue"],

        ml_df["predicted_queue"]
    )


    # ========================================================
    # BASELINE METRICS
    # ========================================================

    baseline_metrics = calculate_metrics(

        baseline_df["actual_queue"],

        baseline_df["predicted_queue"]
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    model_comparison = []

    if MODEL_COMPARISON_FILE.exists():

        comparison_df = pd.read_csv(
            MODEL_COMPARISON_FILE
        )

        model_comparison = (
            comparison_df
            .round(4)
            .to_dict("records")
        )


    # ========================================================
    # CONFUSION MATRIX LABELS
    # ========================================================

    labels = sorted(

        set(
            ml_df["actual_queue"]
        )

        .union(
            ml_df["predicted_queue"]
        )

        .union(
            baseline_df["predicted_queue"]
        )
    )


    # ========================================================
    # ML CONFUSION MATRIX
    # ========================================================

    ml_matrix = confusion_matrix(

        ml_df["actual_queue"],

        ml_df["predicted_queue"],

        labels=labels
    )


    # ========================================================
    # BASELINE CONFUSION MATRIX
    # ========================================================

    baseline_matrix = confusion_matrix(

        baseline_df["actual_queue"],

        baseline_df["predicted_queue"],

        labels=labels
    )


    # ========================================================
    # CONVERT TO LIST
    # ========================================================

    ml_matrix = (
        ml_matrix.tolist()
    )

    baseline_matrix = (
        baseline_matrix.tolist()
    )


    # ========================================================
# AGREEMENT BETWEEN ML AND HEURISTIC MODELS
# ========================================================

    if (
        "ticket_id" in ml_df.columns
        and "ticket_id" in baseline_df.columns
    ):

        comparison_df = pd.merge(
            ml_df[
                [
                    "ticket_id",
                    "predicted_queue"
                ]
            ],
            baseline_df[
                [
                    "ticket_id",
                    "predicted_queue"
                ]
            ],
            on="ticket_id",
            how="inner",
            suffixes=(
                "_ml",
                "_baseline"
            )
        )

        if len(comparison_df) > 0:

            agreement_count = (
                comparison_df["predicted_queue_ml"]
                ==
                comparison_df["predicted_queue_baseline"]
            ).sum()

            agreement_percentage = round(
                agreement_count
                / len(comparison_df)
                * 100,
                2
            )

        else:

            agreement_percentage = 0

    else:

        agreement_percentage = 0

    


    # ========================================================
    # DETERMINE BETTER MODEL
    # ========================================================

    if (
        ml_metrics["f1"]
        >= baseline_metrics["f1"]
    ):

        better_model = (
            "Machine Learning"
        )

    else:

        better_model = (
            "Heuristic Baseline"
        )


    # ========================================================
    # RENDER DASHBOARD
    # ========================================================

    return render_template(

        "dashboard.html",

        ml_metrics=ml_metrics,

        baseline_metrics=baseline_metrics,

        model_comparison=model_comparison,

        labels=labels,

        ml_matrix=ml_matrix,

        baseline_matrix=baseline_matrix,

        agreement_percentage=(
            agreement_percentage
        ),

        better_model=better_model,

        test_size=len(ml_df)

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)