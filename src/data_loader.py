"""
Data Loader
Loads real MovieLens CSV files from data/ directory
Expected files: ratings.csv, movies.csv, tags.csv, links.csv
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = "data"


def load_ratings(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "ratings.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    print(f"[Loader] ratings: {len(df):,} rows | "
          f"{df['userId'].nunique():,} users | "
          f"{df['movieId'].nunique():,} items")
    return df


def load_movies(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "movies.csv")
    df = pd.read_csv(path)
    df["genres_list"] = df["genres"].str.split("|")
    df["genre_count"] = df["genres_list"].apply(len)
    df["has_year"] = df["title"].str.contains(r"\(\d{4}\)", regex=True)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)
    print(f"[Loader] movies: {len(df):,} rows | "
          f"{df['genres'].nunique():,} unique genre combos")
    return df


def load_tags(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "tags.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    print(f"[Loader] tags: {len(df):,} rows | "
          f"{df['userId'].nunique():,} taggers")
    return df


def load_all() -> dict:
    print("[Loader] Loading MovieLens dataset...")
    data = {
        "ratings": load_ratings(),
        "movies":  load_movies(),
        "tags":    load_tags(),
    }
    print(f"[Loader] Dataset loaded successfully\n")
    return data


if __name__ == "__main__":
    data = load_all()
    print("\nRatings sample:")
    print(data["ratings"].head(3).to_string(index=False))
