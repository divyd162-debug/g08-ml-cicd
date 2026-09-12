from pathlib import Path
import pandas as pd

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
# STEP 2: LINEAR SVM C VALUE TUNING
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_FILE = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "customer_support_preprocessed.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 75)
print("G-08 - AUTOMATED CUSTOMER SUPPORT TICKET ROUTING SYSTEM")
print("STEP 2: LINEAR SVM C VALUE TUNING")
print("=" * 75)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not PROCESSED_FILE.exists():

    raise FileNotFoundError(
        f"""
Processed dataset not found:

{PROCESSED_FILE}

Please run:

python src\\preprocess.py

first.
"""
    )


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n[1] Loading processed dataset...")

df = pd.read_csv(PROCESSED_FILE)

print("Dataset shape:", df.shape)

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
# 5. CLEAN DATA
# ============================================================

print("\n[2] Preparing data...")

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

X = df["clean_text"]
y = df["queue"]

print("Final records:", len(df))
print("Number of classes:", y.nunique())

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n[3] Creating train/test split...")

X_train_text, X_test_text, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("Training records:", len(X_train_text))
print("Testing records :", len(X_test_text))


# ============================================================
# 7. WORD TF-IDF
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
    X_train_text
)

X_test_word = word_vectorizer.transform(
    X_test_text
)

print(
    "Word vocabulary:",
    len(word_vectorizer.vocabulary_)
)


# ============================================================
# 8. CHARACTER TF-IDF
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
    X_train_text
)

X_test_char = char_vectorizer.transform(
    X_test_text
)

print(
    "Character vocabulary:",
    len(char_vectorizer.vocabulary_)
)


# ============================================================
# 9. COMBINE FEATURES
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
# 10. C VALUES
# ============================================================

C_VALUES = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0
]


print("\n" + "=" * 75)
print("TESTING DIFFERENT C VALUES")
print("=" * 75)

print("C values:", C_VALUES)


# ============================================================
# 11. TRAIN MODELS
# ============================================================

results = []

best_c = None
best_macro_f1 = -1


for c_value in C_VALUES:

    print("\n" + "-" * 75)

    print(
        f"Training Linear SVM with C = {c_value}"
    )

    print("-" * 75)

    model = LinearSVC(

        C=c_value,

        class_weight="balanced",

        random_state=42,

        max_iter=5000
    )

    model.fit(
        X_train_combined,
        y_train
    )

    predictions = model.predict(
        X_test_combined
    )

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

    print(
        f"Accuracy       : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted Prec. : {precision:.4f}"
    )

    print(
        f"Weighted Recall: {recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : {weighted_f1:.4f}"
    )

    results.append({

        "C": c_value,

        "Accuracy": accuracy,

        "Weighted_Precision": precision,

        "Weighted_Recall": recall,

        "Macro_F1": macro_f1,

        "Weighted_F1": weighted_f1
    })

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1

        best_c = c_value


# ============================================================
# 12. RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 75)
print("C VALUE COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Weighted_Precision": "{:.4f}".format,
            "Weighted_Recall": "{:.4f}".format,
            "Macro_F1": "{:.4f}".format,
            "Weighted_F1": "{:.4f}".format
        }
    )
)


# ============================================================
# 13. BEST C
# ============================================================

print("\n" + "=" * 75)
print("BEST C VALUE")
print("=" * 75)

best_row = results_df.iloc[0]

print(
    f"Best C       : {best_c}"
)

print(
    f"Best Accuracy: "
    f"{best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Best Macro F1: "
    f"{best_row['Macro_F1']:.4f}"
)

print(
    f"Best Macro F1: "
    f"{best_row['Macro_F1'] * 100:.2f}%"
)


# ============================================================
# 14. VERIFY FINAL CHOICE
# ============================================================

if best_c != 7.0:

    print("\nWARNING:")
    print(
        f"The automatically selected C is {best_c}, "
        "not 7."
    )

    print(
        "The training script is configured to use C=7 "
        "as the finalized project model."
    )

else:

    print(
        "\nC=7 is the best-performing value "
        "according to Macro F1."
    )


# ============================================================
# 15. SAVE RESULTS
# ============================================================

tuning_file = (
    RESULTS_DIR
    / "svm_c_tuning_results.csv"
)

results_df.to_csv(
    tuning_file,
    index=False,
    encoding="utf-8"
)

print(
    "\nC tuning results saved to:"
)

print(tuning_file)


# ============================================================
# 16. COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("C VALUE TUNING COMPLETED")
print("=" * 75)

print(
    f"\nRecommended final C: {best_c}"
)

print(
    f"Best Macro F1: "
    f"{best_row['Macro_F1']:.4f}"
)

print("\nNext step:")

print(
    "python src\\train_model.py"
)

print("=" * 75)