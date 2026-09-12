from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
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
# STEP 4: HEURISTIC BASELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "customer_support_preprocessed.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# KEYWORD RULES
# ============================================================

RULES = {

    "Billing and Payments": [
        "payment",
        "billing",
        "charged",
        "charge",
        "invoice",
        "refund",
        "credit card",
        "debit card",
        "transaction",
        "payment failed"
    ],

    "Technical Support": [
        "error",
        "bug",
        "software",
        "application",
        "technical",
        "crash",
        "crashing",
        "technical issue",
        "not working"
    ],

    "IT Support": [
        "vpn",
        "server",
        "network",
        "login",
        "password",
        "computer",
        "access",
        "account access"
    ],

    "Returns and Exchanges": [
        "return",
        "replacement",
        "exchange",
        "returned",
        "replace",
        "return product"
    ],

    "Service Outages and Maintenance": [
        "outage",
        "offline",
        "downtime",
        "maintenance",
        "service unavailable",
        "disruption"
    ],

    "Sales and Pre-Sales": [
        "buy",
        "purchase",
        "pricing",
        "sales",
        "quote",
        "demo",
        "product price"
    ],

    "Product Support": [
        "product",
        "feature",
        "device",
        "product issue",
        "product problem"
    ],

    "Customer Service": [
        "customer service",
        "help",
        "support",
        "question",
        "request"
    ],

    "Human Resources": [
        "employee",
        "hr",
        "human resources",
        "leave",
        "salary",
        "payroll"
    ],

    "General Inquiry": [
        "information",
        "inquiry",
        "general question",
        "more information"
    ]
}


# ============================================================
# HEURISTIC PREDICTION FUNCTION
# ============================================================

def predict_queue(text):

    text = str(text).lower()

    scores = {}

    for queue, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[queue] = score

    # No matching keyword
    if max(scores.values()) == 0:
        return "General Inquiry"

    # Return queue having the highest score
    return max(
        scores,
        key=scores.get
    )


# ============================================================
# RUN BASELINE EVALUATION
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE)

    X = df["clean_text"]
    y = df["queue"]

    # Same split as ML model
    ticket_ids = df.index

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        ticket_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Generate predictions
    heuristic_predictions = X_test.apply(
        predict_queue
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        heuristic_predictions
    )

    precision = precision_score(
        y_test,
        heuristic_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        heuristic_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        heuristic_predictions,
        average="macro",
        zero_division=0
    )

    print("=" * 60)
    print("HEURISTIC BASELINE RESULTS")
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("Macro F1 :", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            heuristic_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results = pd.DataFrame({
    "ticket_id": id_test,
    "ticket_text": X_test,
    "actual_queue": y_test,
    "predicted_queue": heuristic_predictions
    })

    results.to_csv(
        RESULTS_DIR / "heuristic_predictions.csv",
        index=False
    )

    print(
        "\nSaved:",
        RESULTS_DIR / "heuristic_predictions.csv"
    )