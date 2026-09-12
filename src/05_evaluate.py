from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# G-08
# Automated Customer Support Ticket Routing System
#
# STEP 4: FINAL MODEL EVALUATION
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ML_FILE = (
    RESULTS_DIR
    / "best_svm_predictions.csv"
)

HEURISTIC_FILE = (
    RESULTS_DIR
    / "heuristic_predictions.csv"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "model_comparison.csv"
)

CONFUSION_MATRIX_FILE = (
    RESULTS_DIR
    / "confusion_matrix.png"
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 75)
print("G-08 - AUTOMATED CUSTOMER SUPPORT TICKET ROUTING SYSTEM")
print("STEP 4: FINAL MODEL EVALUATION")
print("=" * 75)


# ============================================================
# 3. CHECK ML RESULTS
# ============================================================

if not ML_FILE.exists():

    raise FileNotFoundError(
        f"""
ML prediction file not found:

{ML_FILE}

Please run:

python src\\train_model.py

first.
"""
    )


# ============================================================
# 4. LOAD ML PREDICTIONS
# ============================================================

print("\n[1] Loading ML predictions...")

ml = pd.read_csv(
    ML_FILE
)

print(
    "ML prediction records:",
    len(ml)
)


required_ml_columns = [
    "actual_queue",
    "predicted_queue"
]

missing_ml_columns = [
    column
    for column in required_ml_columns
    if column not in ml.columns
]

if missing_ml_columns:

    raise ValueError(
        f"""
Missing ML columns:

{missing_ml_columns}

Available columns:

{ml.columns.tolist()}
"""
    )


# ============================================================
# 5. ML METRICS
# ============================================================

print("\n[2] Calculating ML metrics...")

ml_accuracy = accuracy_score(

    ml["actual_queue"],

    ml["predicted_queue"]
)


ml_precision = precision_score(

    ml["actual_queue"],

    ml["predicted_queue"],

    average="weighted",

    zero_division=0
)


ml_recall = recall_score(

    ml["actual_queue"],

    ml["predicted_queue"],

    average="weighted",

    zero_division=0
)


ml_macro_f1 = f1_score(

    ml["actual_queue"],

    ml["predicted_queue"],

    average="macro",

    zero_division=0
)


ml_weighted_f1 = f1_score(

    ml["actual_queue"],

    ml["predicted_queue"],

    average="weighted",

    zero_division=0
)


# ============================================================
# 6. DISPLAY ML RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL ML MODEL PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy           : "
    f"{ml_accuracy:.4f} "
    f"({ml_accuracy * 100:.2f}%)"
)

print(
    f"Weighted Precision : "
    f"{ml_precision:.4f}"
)

print(
    f"Weighted Recall    : "
    f"{ml_recall:.4f}"
)

print(
    f"Macro F1           : "
    f"{ml_macro_f1:.4f}"
)

print(
    f"Weighted F1        : "
    f"{ml_weighted_f1:.4f}"
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)

from sklearn.metrics import classification_report

print(
    classification_report(
        ml["actual_queue"],
        ml["predicted_queue"],
        zero_division=0
    )
)


# ============================================================
# 8. HEURISTIC EVALUATION
# ============================================================

heuristic_available = HEURISTIC_FILE.exists()


if heuristic_available:

    print(
        "\n[3] Loading heuristic predictions..."
    )

    heuristic = pd.read_csv(
        HEURISTIC_FILE
    )

    required_heuristic_columns = [
        "actual_queue",
        "predicted_queue"
    ]

    missing_heuristic_columns = [
        column
        for column in required_heuristic_columns
        if column not in heuristic.columns
    ]

    if missing_heuristic_columns:

        raise ValueError(
            f"""
Missing heuristic columns:

{missing_heuristic_columns}

Available columns:

{heuristic.columns.tolist()}
"""
        )


    h_accuracy = accuracy_score(

        heuristic["actual_queue"],

        heuristic["predicted_queue"]
    )


    h_precision = precision_score(

        heuristic["actual_queue"],

        heuristic["predicted_queue"],

        average="weighted",

        zero_division=0
    )


    h_recall = recall_score(

        heuristic["actual_queue"],

        heuristic["predicted_queue"],

        average="weighted",

        zero_division=0
    )


    h_macro_f1 = f1_score(

        heuristic["actual_queue"],

        heuristic["predicted_queue"],

        average="macro",

        zero_division=0
    )


    print(
        "\nHeuristic Accuracy:",
        f"{h_accuracy * 100:.2f}%"
    )

    print(
        "Heuristic Macro F1:",
        f"{h_macro_f1:.4f}"
    )


else:

    print(
        "\n[3] Heuristic prediction file not found."
    )

    print(
        "Baseline comparison will be skipped."
    )

    h_accuracy = None
    h_precision = None
    h_recall = None
    h_macro_f1 = None


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("MODEL COMPARISON")
print("=" * 75)


if heuristic_available:

    comparison = pd.DataFrame({

        "Model": [
            "ML Classifier",
            "Keyword Heuristic"
        ],

        "Accuracy": [
            ml_accuracy,
            h_accuracy
        ],

        "Precision": [
            ml_precision,
            h_precision
        ],

        "Recall": [
            ml_recall,
            h_recall
        ],

        "Macro F1": [
            ml_macro_f1,
            h_macro_f1
        ]

    })

else:

    comparison = pd.DataFrame({

        "Model": [
            "ML Classifier"
        ],

        "Accuracy": [
            ml_accuracy
        ],

        "Precision": [
            ml_precision
        ],

        "Recall": [
            ml_recall
        ],

        "Macro F1": [
            ml_macro_f1
        ]

    })


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE COMPARISON
# ============================================================

comparison.to_csv(

    COMPARISON_FILE,

    index=False,

    encoding="utf-8"
)


print(
    "\nComparison saved to:"
)

print(
    COMPARISON_FILE
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n[4] Creating confusion matrix...")


labels = sorted(
    set(ml["actual_queue"])
    | set(ml["predicted_queue"])
)


cm = confusion_matrix(

    ml["actual_queue"],

    ml["predicted_queue"],

    labels=labels
)


fig, ax = plt.subplots(
    figsize=(16, 13)
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=labels
)


disp.plot(

    ax=ax,

    cmap="Blues",

    values_format="d",

    xticks_rotation=45,

    colorbar=True
)


ax.set_title(
    "Confusion Matrix - Final C=7 ML Ticket Routing Model",
    fontsize=18,
    pad=20
)


ax.set_xlabel(
    "Predicted Queue",
    fontsize=13
)


ax.set_ylabel(
    "Actual Queue",
    fontsize=13
)


plt.xticks(
    fontsize=10,
    ha="right"
)

plt.yticks(
    fontsize=10
)


plt.tight_layout()


plt.savefig(

    CONFUSION_MATRIX_FILE,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nConfusion matrix saved to:"
)

print(
    CONFUSION_MATRIX_FILE
)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 EVALUATION COMPLETED")
print("=" * 75)

print(
    f"""
Final Model: Linear SVM
C Value: 7

Accuracy           : {ml_accuracy * 100:.2f}%
Weighted Precision : {ml_precision:.4f}
Weighted Recall    : {ml_recall:.4f}
Macro F1           : {ml_macro_f1:.4f}
Weighted F1        : {ml_weighted_f1:.4f}
"""
)

print(
    "Generated files:"
)

print(
    "1.",
    COMPARISON_FILE
)

print(
    "2.",
    CONFUSION_MATRIX_FILE
)

print("\n" + "=" * 75)
print("EVALUATION FINISHED")
print("=" * 75)