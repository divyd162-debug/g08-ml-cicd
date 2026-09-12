import pandas as pd
import re
from pathlib import Path

# ============================================================
# G-08
# Automated Customer Support Ticket Routing System
# STEP 2: Data Preprocessing
# ============================================================

# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "dataset"
    / "raw"
    / "dataset-tickets-expanded.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "customer_support_preprocessed.csv"
)

# Create processed folder
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load original dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset:", df.shape)

# ------------------------------------------------------------
# 3. Check required columns
# ------------------------------------------------------------

required_columns = [
    "language",
    "subject",
    "body",
    "queue"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. Select English tickets
# ------------------------------------------------------------

df["language"] = df["language"].astype(str).str.lower().str.strip()

df = df[
    df["language"] == "en"
].copy()

print("English tickets:", df.shape)

# ------------------------------------------------------------
# 5. Select required columns
# ------------------------------------------------------------

df = df[
    [
        "subject",
        "body",
        "queue"
    ]
].copy()

# ------------------------------------------------------------
# 6. Check missing values
# ------------------------------------------------------------

print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 7. Handle missing values
# ------------------------------------------------------------

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["queue"] = df["queue"].fillna("")

# Convert to string
df["subject"] = df["subject"].astype(str)
df["body"] = df["body"].astype(str)
df["queue"] = df["queue"].astype(str).str.strip()

# ------------------------------------------------------------
# 8. Combine subject and body
# ------------------------------------------------------------

df["ticket_text"] = (
    df["subject"].str.strip()
    + " "
    + df["body"].str.strip()
).str.strip()

# ------------------------------------------------------------
# 9. Basic text cleaning
# ------------------------------------------------------------

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


df["clean_text"] = df["ticket_text"].apply(clean_text)

# ------------------------------------------------------------
# 10. Remove empty ticket text
# ------------------------------------------------------------

before = len(df)

df = df[
    df["clean_text"].str.len() > 0
].copy()

after = len(df)

print("\nEmpty text rows removed:", before - after)

# ------------------------------------------------------------
# 11. Remove rows without a queue/category
# ------------------------------------------------------------

before = len(df)

df = df[
    df["queue"].str.len() > 0
].copy()

after = len(df)

print("Rows without queue removed:", before - after)

# ------------------------------------------------------------
# 12. Remove duplicate tickets
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=["clean_text", "queue"]
).copy()

after = len(df)

print("Duplicate tickets removed:", before - after)

# ------------------------------------------------------------
# 13. Final missing-value check
# ------------------------------------------------------------

print("\nMissing values after preprocessing:")

print(
    df[
        [
            "subject",
            "body",
            "queue",
            "clean_text"
        ]
    ].isnull().sum()
)

# ------------------------------------------------------------
# 14. Queue distribution
# ------------------------------------------------------------

print("\nQueue distribution:")
print(df["queue"].value_counts())

# ------------------------------------------------------------
# 15. Number of categories
# ------------------------------------------------------------

print(
    "\nNumber of unique queues:",
    df["queue"].nunique()
)

# ------------------------------------------------------------
# 16. Show sample processed tickets
# ------------------------------------------------------------

print("\nSample processed tickets:")

print(
    df[
        [
            "subject",
            "body",
            "queue",
            "clean_text"
        ]
    ]
    .head(5)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 17. Save processed dataset
# ------------------------------------------------------------

df[
    [
        "subject",
        "body",
        "queue",
        "clean_text"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# ------------------------------------------------------------
# 18. Final information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Processed dataset saved at:")
print(OUTPUT_FILE)

print("\nFinal dataset shape:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("=" * 60)