
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "processed"

MODEL_PATH = DATA_DIR / "svd_model.pkl"
MOVIES_PATH = DATA_DIR / "movies_catalog.csv"
TRAIN_PATH = DATA_DIR / "train_split.csv"

# Check that the training artifacts exist
for path in [MODEL_PATH, MOVIES_PATH, TRAIN_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

# Load the saved model and data
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

movies = pd.read_csv(MOVIES_PATH)
train_df = pd.read_csv(TRAIN_PATH)

# Keep IDs consistent
movies["movie_id"] = movies["movie_id"].astype(int)
train_df["user_id"] = train_df["user_id"].astype(int)
train_df["movie_id"] = train_df["movie_id"].astype(int)

app = FastAPI(
    title="CineTale Recommendation API",
    version="1.0.0"
)

# Allow requests from the local React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "CineTale Recommendation API is running!"
    }


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    limit: int = Query(default=10, ge=1, le=50)
):
    known_users = set(train_df["user_id"].unique())

    if user_id not in known_users:
        raise HTTPException(
            status_code=404,
            detail="User not found in training data."
        )

    # Exclude movies the user rated during training
    seen_movies = set(
        train_df.loc[
            train_df["user_id"] == user_id,
            "movie_id"
        ]
    )

    candidates = movies[
        ~movies["movie_id"].isin(seen_movies)
    ].copy()

    if candidates.empty:
        return {"user_id": user_id, "recommendations": []}

    # Predict ratings for unseen movies
    candidates["predicted_rating"] = [
        model.predict(user_id, int(movie_id)).est
        for movie_id in candidates["movie_id"]
    ]

    candidates = candidates.sort_values(
        "predicted_rating",
        ascending=False
    ).head(limit)

    results = candidates.fillna("").to_dict(
        orient="records"
    )

    return {
        "user_id": user_id,
        "recommendations": results
    }