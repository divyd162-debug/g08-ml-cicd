from pathlib import Path
import pandas as pd
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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
# STEP 3: FINAL C=3 LINEAR SVM MODEL TRAINING
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "customer_support_preprocessed.csv"
)

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

FINAL_C = 3.0

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# 3. OUTPUT FILES
# ============================================================

WORD_VECTORIZER_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer_best_svm.pkl"
)

CHAR_VECTORIZER_PATH = (
    MODEL_DIR
    / "char_tfidf_vectorizer_best_svm.pkl"
)

MODEL_PATH = (
    MODEL_DIR
    / "ticket_classifier_best_svm.pkl"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "best_svm_predictions.csv"
)

TRAINING_INFO_PATH = (
    RESULTS_DIR
    / "best_svm_training_info.csv"
)


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 75)
print("G-08 - AUTOMATED CUSTOMER SUPPORT TICKET ROUTING SYSTEM")
print("STEP 3: FINAL C=3 LINEAR SVM MODEL TRAINING")
print("=" * 75)


# ============================================================
# 5. CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"""
Processed dataset not found:

{DATA_PATH}

Please run:

python src\\preprocess.py

first.
"""
    )


# ============================================================
# 6. LOAD DATASET
# ============================================================

print("\n[1] Loading processed dataset...")

df = pd.read_csv(DATA_PATH)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 7. VALIDATE COLUMNS
# ============================================================

required_columns = [
    "clean_text",
    "queue"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"""
Missing required columns:

{missing_columns}

Available columns:

{df.columns.tolist()}
"""
    )


# ============================================================
# 8. CLEAN DATA
# ============================================================

print("\n[2] Validating data...")

df["clean_text"] = (
    df["clean_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["queue"] = (
    df["queue"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    (df["clean_text"] != "") &
    (df["queue"] != "")
].copy()

df = df.drop_duplicates(
    subset=[
        "clean_text",
        "queue"
    ]
)

df = df.reset_index(drop=True)


print(
    "Final records:",
    len(df)
)

print(
    "Number of classes:",
    df["queue"].nunique()
)


# ============================================================
# 9. FEATURES AND TARGET
# ============================================================

X = df["clean_text"]
y = df["queue"]


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Creating train/test split...")

ticket_ids = df.index

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    ticket_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records :",
    len(X_test)
)


# ============================================================
# 11. WORD TF-IDF
# ============================================================

print("\n[4] Creating WORD-level TF-IDF...")

word_vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=1,

    max_df=0.98,

    max_features=30000,

    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(
    X_train
)

X_test_word = word_vectorizer.transform(
    X_test
)

print(
    "Word vocabulary:",
    len(word_vectorizer.vocabulary_)
)

print(
    "Word feature shape:",
    X_train_word.shape
)


# ============================================================
# 12. CHARACTER TF-IDF
# ============================================================

print("\n[5] Creating CHARACTER-level TF-IDF...")

char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3, 5),

    min_df=2,

    max_df=0.98,

    max_features=30000,

    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(
    X_train
)

X_test_char = char_vectorizer.transform(
    X_test
)

print(
    "Character vocabulary:",
    len(char_vectorizer.vocabulary_)
)

print(
    "Character feature shape:",
    X_train_char.shape
)


# ============================================================
# 13. COMBINE FEATURES
# ============================================================

print("\n[6] Combining word + character features...")

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
    "Combined training shape:",
    X_train_combined.shape
)

print(
    "Combined testing shape:",
    X_test_combined.shape
)


# ============================================================
# 14. TRAIN FINAL C=3 SVM
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL LINEAR SVM")
print("=" * 75)

print(
    f"C value: {FINAL_C}"
)

print(
    "Class weight: balanced"
)

print(
    "Maximum iterations: 5000"
)


model = LinearSVC(

    C=FINAL_C,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    max_iter=5000
)


model.fit(
    X_train_combined,
    y_train
)


print(
    "\nFinal C=3 SVM training completed."
)


# ============================================================
# 15. PREDICTIONS
# ============================================================

print("\n[7] Generating predictions...")

predictions = model.predict(
    X_test_combined
)


# ============================================================
# 16. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# 17. DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("FINAL MODEL PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy           : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Weighted Precision : {precision:.4f}"
)

print(
    f"Weighted Recall    : {recall:.4f}"
)

print(
    f"Macro F1           : {macro_f1:.4f}"
)

print(
    f"Weighted F1        : {weighted_f1:.4f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 19. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "ticket_id":
        id_test.values,

    "ticket_text":
        X_test.values,

    "actual_queue":
        y_test.values,

    "predicted_queue":
        predictions
})


prediction_df[
    "correct_prediction"
] = (

    prediction_df[
        "actual_queue"
    ]

    ==

    prediction_df[
        "predicted_queue"
    ]
)


prediction_df.to_csv(

    PREDICTIONS_PATH,

    index=False,

    encoding="utf-8"
)


print(
    "\nPredictions saved to:"
)

print(
    PREDICTIONS_PATH
)


# ============================================================
# 20. SAVE WORD VECTORIZER
# ============================================================

joblib.dump(

    word_vectorizer,

    WORD_VECTORIZER_PATH
)


print(
    "\nWord TF-IDF vectorizer saved to:"
)

print(
    WORD_VECTORIZER_PATH
)


# ============================================================
# 21. SAVE CHARACTER VECTORIZER
# ============================================================

joblib.dump(

    char_vectorizer,

    CHAR_VECTORIZER_PATH
)


print(
    "\nCharacter TF-IDF vectorizer saved to:"
)

print(
    CHAR_VECTORIZER_PATH
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH
)


print(
    "\nFinal C=3 SVM model saved to:"
)

print(
    MODEL_PATH
)


# ============================================================
# 23. SAVE TRAINING INFORMATION
# ============================================================

training_info = pd.DataFrame([{

    "model":
        "LinearSVC",

    "C":
        FINAL_C,

    "class_weight":
        "balanced",

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "number_of_classes":
        y.nunique(),

    "train_samples":
        len(X_train),

    "test_samples":
        len(X_test),

    "word_features":
        X_train_word.shape[1],

    "character_features":
        X_train_char.shape[1],

    "total_features":
        X_train_combined.shape[1],

    "accuracy":
        accuracy,

    "weighted_precision":
        precision,

    "weighted_recall":
        recall,

    "macro_f1":
        macro_f1,

    "weighted_f1":
        weighted_f1
}])


training_info.to_csv(

    TRAINING_INFO_PATH,

    index=False,

    encoding="utf-8"
)


print(
    "\nTraining information saved to:"
)

print(
    TRAINING_INFO_PATH
)


# ============================================================
# 24. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nFinal Model: Linear SVM"
)

print(
    f"C Value: {FINAL_C}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)

print("\nGenerated files:")

print(
    "1.",
    MODEL_PATH
)

print(
    "2.",
    WORD_VECTORIZER_PATH
)

print(
    "3.",
    CHAR_VECTORIZER_PATH
)

print(
    "4.",
    PREDICTIONS_PATH
)

print(
    "5.",
    TRAINING_INFO_PATH
)

print("\n" + "=" * 75)
print("FINAL c=3 MODEL TRAINING FINISHED")
print("=" * 75)