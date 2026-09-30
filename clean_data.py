
import pandas as pd
from pathlib import Path

# File locations
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "train.csv"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load original data
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# 2. Correct the misleading column names
df = df.rename(columns={
    "release_date": "genres",
    "sex": "age",
    "age": "gender"
})

# 3. Remove exact duplicate rows
df = df.drop_duplicates()

# 4. Standardize text columns
for col in ["title", "genres", "gender"]:
    df[col] = df[col].astype("string").str.strip()

# 5. Ensure numeric columns have valid types
for col in ["user_id", "movie_id", "rating", "age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Remove rows with missing required values
required = [
    "user_id", "title", "movie_id",
    "rating", "genres", "age", "gender"
]
df = df.dropna(subset=required)

# 7. Validate IDs and ratings
df = df[
    (df["user_id"] > 0)
    & (df["movie_id"] > 0)
    & (df["rating"].between(1, 5))
    & (df["user_id"] % 1 == 0)
    & (df["movie_id"] % 1 == 0)
    & (df["age"] % 1 == 0)
]

# 8. Remove empty titles and genres
df = df[
    df["title"].ne("")
    & df["genres"].ne("")
]

# 9. Normalize gender values
df["gender"] = df["gender"].str.upper()

df = df[df["gender"].isin(["M", "F"])]

# 10. Convert ID and demographic columns
df["user_id"] = df["user_id"].astype("int64")
df["movie_id"] = df["movie_id"].astype("int64")
df["age"] = df["age"].astype("int64")

# 11. Check repeated user-movie interactions
pair_duplicates = df.duplicated(
    subset=["user_id", "movie_id"]
).sum()

print("\nRepeated user-movie pairs:", pair_duplicates)

# Keep these records for now; investigate before removing.
# Different ratings for the same movie may carry useful history.

# 12. Save cleaned data
output_file = OUTPUT_DIR / "train_clean.csv"
df.to_csv(output_file, index=False)

# 13. Summary
print("\nCleaning completed!")
print("Clean shape:", df.shape)
print("Unique users:", df["user_id"].nunique())
print("Unique movies:", df["movie_id"].nunique())
print("Rating distribution:")
print(df["rating"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSaved to:", output_file)