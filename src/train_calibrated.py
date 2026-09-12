from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# G-08
# Automated Customer Support Ticket Routing System
#
# CALIBRATED C=3 SVM
#
# Purpose:
# Generate probability estimates for the Flask web application.
#
# Model:
# LinearSVC
# C = 3
# class_weight = balanced
#
# Features:
# Word TF-IDF + Character TF-IDF
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

# Automatically finds the project root.
# Works on both Windows and Mac.

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "customer_support_preprocessed.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)


# Create folders if they do not exist

MODEL_DIR.mkdir(
    exist_ok=True
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

WORD_TFIDF_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer_calibrated.pkl"
)

CHAR_TFIDF_FILE = (
    MODEL_DIR
    / "char_tfidf_vectorizer_calibrated.pkl"
)

MODEL_FILE = (
    MODEL_DIR
    / "ticket_classifier_calibrated.pkl"
)

INFO_FILE = (
    RESULTS_DIR
    / "calibrated_training_info.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

C_VALUE = 3


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)

print(
    "G-08 AUTOMATED CUSTOMER SUPPORT "
    "TICKET ROUTING SYSTEM"
)

print("=" * 70)

print("\nLoading processed dataset...")


if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
)


print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

# preprocess.py creates clean_text and queue

required_columns = [
    "clean_text",
    "queue"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.dropna(
    subset=[
        "clean_text",
        "queue"
    ]
).copy()


df["clean_text"] = (
    df["clean_text"]
    .astype(str)
    .str.strip()
)


df["queue"] = (
    df["queue"]
    .astype(str)
    .str.strip()
)


# Remove empty text

df = df[
    df["clean_text"] != ""
]


# Remove empty labels

df = df[
    df["queue"] != ""
]


print(
    f"Usable rows: {len(df)}"
)


# ============================================================
# INPUT AND TARGET
# ============================================================

X = df["clean_text"]

y = df["queue"]


print(
    f"Number of classes: {y.nunique()}"
)


print("\nClass distribution:")

print(
    y.value_counts()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# WORD TF-IDF
# ============================================================

print("\nCreating Word TF-IDF...")


word_tfidf = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.98,

    max_features=30000,

    sublinear_tf=True
)


X_train_word = word_tfidf.fit_transform(
    X_train
)


X_test_word = word_tfidf.transform(
    X_test
)


print(
    "Word features:",
    len(word_tfidf.vocabulary_)
)


# ============================================================
# CHARACTER TF-IDF
# ============================================================

print("\nCreating Character TF-IDF...")


char_tfidf = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3, 5),

    min_df=2,

    max_df=0.98,

    max_features=30000,

    sublinear_tf=True
)


X_train_char = char_tfidf.fit_transform(
    X_train
)


X_test_char = char_tfidf.transform(
    X_test
)


print(
    "Character features:",
    len(char_tfidf.vocabulary_)
)


# ============================================================
# COMBINE FEATURES
# ============================================================

print(
    "\nCombining Word + Character features..."
)


X_train_combined = hstack(
    [
        X_train_word,
        X_train_char
    ]
).tocsr()


X_test_combined = hstack(
    [
        X_test_word,
        X_test_char
    ]
).tocsr()


print(
    "Total features:",
    X_train_combined.shape[1]
)


# ============================================================
# BASE LINEAR SVM
# ============================================================

print("\nCreating LinearSVC...")


base_svm = LinearSVC(

    C=C_VALUE,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    max_iter=5000
)


# ============================================================
# CALIBRATED CLASSIFIER
# ============================================================

print("\nCreating calibrated C=3 SVM...")

print(
    "Calibration method: sigmoid"
)

print(
    "Calibration folds: 5"
)


# For newer versions of scikit-learn

calibrated_model = (
    CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=5
    )
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining calibrated model...")


calibrated_model.fit(
    X_train_combined,
    y_train
)


print(
    "\nTraining completed successfully."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nEvaluating calibrated model..."
)


y_pred = calibrated_model.predict(
    X_test_combined
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


print(
    "\n" + "=" * 70
)

print(
    "CALIBRATED C=3 MODEL RESULTS"
)

print(
    "=" * 70
)


print(
    f"Accuracy : {accuracy * 100:.2f}%"
)


print(
    f"Precision: {precision * 100:.2f}%"
)


print(
    f"Recall   : {recall * 100:.2f}%"
)


print(
    f"Macro F1 : {f1 * 100:.2f}%"
)


print(
    "=" * 70
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:\n"
)


print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CHECK PROBABILITIES
# ============================================================

print(
    "\nChecking probability output..."
)


probabilities = calibrated_model.predict_proba(
    X_test_combined
)


print(
    "Probability matrix shape:",
    probabilities.shape
)


print(
    "Probability range:",
    probabilities.min(),
    "to",
    probabilities.max()
)


# ============================================================
# EXAMPLE PROBABILITY
# ============================================================

classes = calibrated_model.classes_


example_index = 0


example_text = X_test.iloc[
    example_index
]


example_prediction = y_pred[
    example_index
]


example_probabilities = probabilities[
    example_index
]


print(
    "\nExample ticket:"
)


print(
    example_text[:300]
)


print(
    "\nPredicted department:",
    example_prediction
)


print(
    "\nTop probability estimates:"
)


sorted_indices = np.argsort(
    example_probabilities
)[::-1]


for index in sorted_indices[:5]:

    print(
        f"{classes[index]}: "
        f"{example_probabilities[index] * 100:.2f}%"
    )


# ============================================================
# SAVE WORD TF-IDF
# ============================================================

print(
    "\nSaving Word TF-IDF..."
)


joblib.dump(
    word_tfidf,
    WORD_TFIDF_FILE
)


# ============================================================
# SAVE CHARACTER TF-IDF
# ============================================================

print(
    "Saving Character TF-IDF..."
)


joblib.dump(
    char_tfidf,
    CHAR_TFIDF_FILE
)


# ============================================================
# SAVE CALIBRATED MODEL
# ============================================================

print(
    "Saving calibrated model..."
)


joblib.dump(
    calibrated_model,
    MODEL_FILE
)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

training_info = pd.DataFrame({

    "parameter": [

        "model",

        "C",

        "class_weight",

        "calibration_method",

        "calibration_cv",

        "word_features",

        "character_features",

        "total_features",

        "training_samples",

        "testing_samples",

        "accuracy",

        "precision_weighted",

        "recall_weighted",

        "f1_macro"
    ],

    "value": [

        "LinearSVC + CalibratedClassifierCV",

        C_VALUE,

        "balanced",

        "sigmoid",

        5,

        len(word_tfidf.vocabulary_),

        len(char_tfidf.vocabulary_),

        X_train_combined.shape[1],

        len(X_train),

        len(X_test),

        accuracy,

        precision,

        recall,

        f1
    ]
})


training_info.to_csv(
    INFO_FILE,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)


print(
    "CALIBRATED C=3 MODEL SAVED SUCCESSFULLY"
)


print(
    "=" * 70
)


print(
    f"\nWord TF-IDF:\n{WORD_TFIDF_FILE}"
)


print(
    f"\nCharacter TF-IDF:\n{CHAR_TFIDF_FILE}"
)


print(
    f"\nCalibrated Model:\n{MODEL_FILE}"
)


print(
    f"\nTraining Info:\n{INFO_FILE}"
)


print(
    "\nReady for Flask application."
)


print(
    "=" * 70
)