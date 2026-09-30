
import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader, SVD, accuracy

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed"

# Load split datasets
train_df = pd.read_csv(DATA_DIR / "train_split.csv")
test_df = pd.read_csv(DATA_DIR / "test_split.csv")

# Keep only columns needed for collaborative filtering
train_ratings = train_df[["user_id", "movie_id", "rating"]].copy()
test_ratings = test_df[["user_id", "movie_id", "rating"]].copy()

# Surprise expects user ID, item ID, rating
reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(
    train_ratings[["user_id", "movie_id", "rating"]],
    reader
)

trainset = train_data.build_full_trainset()

# Train the SVD collaborative filtering model
model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)

print("Training CineTale recommendation model...")
model.fit(trainset)

print("Training completed!")

# Evaluate on held-out ratings
testset = list(
    test_ratings[
        ["user_id", "movie_id", "rating"]
    ].itertuples(index=False, name=None)
)

predictions = model.test(testset)

print("\nModel evaluation:")
accuracy.rmse(predictions, verbose=True)
accuracy.mae(predictions, verbose=True)

# Load movie titles
movies = (
    train_df[["movie_id", "title"]]
    .drop_duplicates(subset="movie_id")
)

# Select a user for a sample recommendation
user_id = int(train_df["user_id"].iloc[0])

# Movies already rated by this user in training data
seen_movies = set(
    train_ratings.loc[
        train_ratings["user_id"] == user_id,
        "movie_id"
    ]
)

# Score movies the user hasn't rated
candidate_movies = movies[
    ~movies["movie_id"].isin(seen_movies)
].copy()

candidate_movies["predicted_rating"] = candidate_movies[
    "movie_id"
].apply(
    lambda movie_id: model.predict(
        user_id, int(movie_id)
    ).est
)

recommendations = candidate_movies.sort_values(
    "predicted_rating", ascending=False
).head(10)

print(f"\nTop 10 recommendations for user {user_id}:")

print(
    recommendations[
        ["title", "predicted_rating"]
    ].to_string(index=False)
)

# Save sample recommendations
recommendations.to_csv(
    DATA_DIR / "sample_recommendations.csv",
    index=False
)

print("\nSaved sample_recommendations.csv")

import pickle

# Save the trained SVD model
model_path = DATA_DIR / "svd_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

# Save the movie catalog for future recommendations
catalog = (
    train_df[["movie_id", "title", "genres"]]
    .drop_duplicates(subset="movie_id")
)

catalog.to_csv(
    DATA_DIR / "movies_catalog.csv",
    index=False
)

print("Model saved to:", model_path)
print("Movie catalog saved successfully!")