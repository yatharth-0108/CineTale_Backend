
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "processed" / "train_clean.csv"
OUTPUT_DIR = BASE_DIR / "processed"

df = pd.read_csv(INPUT_FILE)

# Shuffle each user's ratings independently.
# Keep at least one rating per user in training.
df = df.sample(frac=1, random_state=42)

train_parts = []
test_parts = []

for user_id, group in df.groupby("user_id", sort=False):
    n = len(group)

    # Users with only one rating stay in training.
    if n < 2:
        train_parts.append(group)
        continue

    n_test = max(1, round(n * 0.2))
    n_test = min(n_test, n - 1)

    test_parts.append(group.iloc[:n_test])
    train_parts.append(group.iloc[n_test:])

train_df = pd.concat(train_parts).sample(
    frac=1, random_state=42
)
test_df = pd.concat(test_parts) if test_parts else df.iloc[0:0]

train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training users:", train_df["user_id"].nunique())
print("Testing users:", test_df["user_id"].nunique())
print("Files saved successfully.")