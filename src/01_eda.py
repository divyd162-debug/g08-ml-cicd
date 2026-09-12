from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# G-08
# Automated Customer Support Ticket Routing System
# STEP 1: EXPLORATORY DATA ANALYSIS
# ============================================================


# ============================================================
# 0. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FILE_PATH = (
    BASE_DIR
    / "dataset"
    / "raw"
    / "dataset-tickets-multi-lang-5-2-50-version.csv"
)


# ============================================================
# 1. CHECK DATASET
# ============================================================

if not FILE_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found!\n\n"
        f"Expected location:\n{FILE_PATH}\n\n"
        f"Please check the filename inside:\n"
        f"{BASE_DIR / 'dataset' / 'raw'}"
    )


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(FILE_PATH)


print("=" * 60)
print("G-08 AUTOMATED CUSTOMER SUPPORT TICKET ROUTING SYSTEM")
print("STEP 1: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset path:")
print(FILE_PATH)

print("\nDataset shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 3. COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

print(df.columns.tolist())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "queue",
    "language",
    "type",
    "priority",
    "subject",
    "body"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"\nMissing required columns: {missing_columns}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ============================================================
# 5. FIRST FIVE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 8. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate records:", duplicate_count)


# ============================================================
# 9. QUEUE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("QUEUE DISTRIBUTION")
print("=" * 60)

queue_distribution = df["queue"].value_counts()

print(queue_distribution)


# ============================================================
# 10. LANGUAGE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("LANGUAGE DISTRIBUTION")
print("=" * 60)

language_distribution = df["language"].value_counts()

print(language_distribution)


# ============================================================
# 11. TICKET TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TICKET TYPE DISTRIBUTION")
print("=" * 60)

type_distribution = df["type"].value_counts()

print(type_distribution)


# ============================================================
# 12. PRIORITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PRIORITY DISTRIBUTION")
print("=" * 60)

priority_distribution = df["priority"].value_counts()

print(priority_distribution)


# ============================================================
# 13. TEXT STATISTICS
# ============================================================

df["subject_length"] = (
    df["subject"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["body_length"] = (
    df["body"]
    .fillna("")
    .astype(str)
    .str.len()
)


print("\n" + "=" * 60)
print("TEXT STATISTICS")
print("=" * 60)

print("\nSubject length:")
print(df["subject_length"].describe())

print("\nBody length:")
print(df["body_length"].describe())


# ============================================================
# 14. QUEUE VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

queue_distribution.sort_values().plot(
    kind="barh"
)

plt.title("Customer Support Tickets by Queue")
plt.xlabel("Number of Tickets")
plt.ylabel("Queue")

plt.tight_layout()
plt.show()


# ============================================================
# 15. LANGUAGE VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

language_distribution.plot(
    kind="bar"
)

plt.title("Customer Support Tickets by Language")
plt.xlabel("Language")
plt.ylabel("Number of Tickets")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 16. PRIORITY VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

priority_distribution.plot(
    kind="bar"
)

plt.title("Customer Support Tickets by Priority")
plt.xlabel("Priority")
plt.ylabel("Number of Tickets")

plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 17. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)