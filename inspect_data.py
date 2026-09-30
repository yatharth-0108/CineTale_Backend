
import pandas as pd

# Inspect the training dataset
train = pd.read_csv("train.csv")

print("===== TRAIN DATASET =====")
print("Shape:", train.shape)

print("\nColumn names:")
print(train.columns.tolist())

print("\nFirst 5 rows:")
print(train.head())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())

print("\nDuplicate rows:", train.duplicated().sum())


# Inspect the baseline dataset
baseline = pd.read_csv("kaggle_baseline.csv")

print("\n===== BASELINE DATASET =====")
print("Shape:", baseline.shape)

print("\nColumn names:")
print(baseline.columns.tolist())

print("\nFirst 5 rows:")
print(baseline.head())