"""
Module 1: User Engagement Profiler
Analyzes behavioral signals: rating volume, rating patterns,
genre diversity, engagement consistency, activity recency
"""
import pandas as pd
import numpy as np
from scipy import stats


def compute_user_profiles(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-user behavioral profile from rating signals.
    Equivalent to analyzing profile completion + engagement depth
    in a social/discovery product.
    """
    print("[EngagementProfiler] Computing user behavioral profiles...")

    # Merge with movie metadata
    merged = ratings.merge(movies[["movieId","genres","genre_count","year"]], on="movieId", how="left")
    merged["genres_list"] = merged["genres"].str.split("|")

    # ── Core engagement signals ──────────────────────────────────
    user_stats = ratings.groupby("userId").agg(
        rating_count        = ("rating", "count"),
        avg_rating          = ("rating", "mean"),
        rating_std          = ("rating", "std"),
        min_rating          = ("rating", "min"),
        max_rating          = ("rating", "max"),
        rating_range        = ("rating", lambda x: x.max() - x.min()),
        first_activity      = ("timestamp", "min"),
        last_activity       = ("timestamp", "max"),
    ).reset_index()

    # ── Genre diversity (breadth of interest) ───────────────────
    def genre_diversity(user_df):
        all_genres = []
        for genres in user_df["genres_list"].dropna():
            all_genres.extend(genres)
        unique = len(set(all_genres))
        total  = len(all_genres)
        # Shannon entropy of genre distribution
        if total == 0:
            return 0.0
        from collections import Counter
        counts = Counter(all_genres)
        probs  = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        return round(float(entropy), 4)

    genre_div = merged.groupby("userId").apply(genre_diversity).reset_index()
    genre_div.columns = ["userId", "genre_entropy"]

    # ── Activity consistency (temporal engagement) ───────────────
    def activity_consistency(user_df):
        if len(user_df) < 3:
            return 0.0
        sorted_ts = user_df["timestamp"].sort_values()
        gaps = sorted_ts.diff().dt.days.dropna()
        if gaps.std() == 0:
            return 1.0
        cv = gaps.std() / (gaps.mean() + 1e-9)
        return round(float(1 / (1 + cv)), 4)

    consistency = ratings.groupby("userId").apply(activity_consistency).reset_index()
    consistency.columns = ["userId", "activity_consistency"]

    # ── Rating discrimination (does user differentiate?) ────────
    user_stats["rating_discrimination"] = (
        user_stats["rating_std"].fillna(0) *
        user_stats["rating_range"]
    ).round(4)

    # ── Session recency (days since last activity) ───────────────
    max_date = ratings["timestamp"].max()
    user_stats["days_since_last"] = (
        max_date - user_stats["last_activity"]
    ).dt.days

    # ── Activity span (days from first to last) ──────────────────
    user_stats["activity_span_days"] = (
        user_stats["last_activity"] - user_stats["first_activity"]
    ).dt.days

    # ── Combine all signals ──────────────────────────────────────
    profiles = user_stats.merge(genre_div,   on="userId", how="left")
    profiles = profiles.merge(consistency,   on="userId", how="left")

    # Fill NaN std for users with 1 rating
    profiles["rating_std"] = profiles["rating_std"].fillna(0)

    print(f"[EngagementProfiler] {len(profiles):,} user profiles computed")
    print(f"  Avg ratings per user:   {profiles['rating_count'].mean():.1f}")
    print(f"  Avg genre entropy:      {profiles['genre_entropy'].mean():.3f}")
    print(f"  Avg activity span:      {profiles['activity_span_days'].mean():.0f} days")

    return profiles


def score_profile_quality(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight profile quality scorer.
    Higher score = richer behavioral signal = more reliable for recommendations.

    Components (all normalized 0-1):
    - Volume score:      how many ratings (depth of engagement)
    - Diversity score:   genre entropy (breadth of interest)
    - Consistency score: temporal regularity
    - Discrimination:    whether user differentiates between good/bad
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    features = ["rating_count", "genre_entropy", "activity_consistency", "rating_discrimination"]
    weights  = [0.35,           0.25,            0.20,                  0.20]

    df = profiles.copy()
    for feat in features:
        df[f"{feat}_norm"] = scaler.fit_transform(df[[feat]])

    df["profile_quality_score"] = sum(
        df[f"{feat}_norm"] * w
        for feat, w in zip(features, weights)
    )
    df["profile_quality_score"] = df["profile_quality_score"].round(4)

    # Tier classification
    df["quality_tier"] = pd.cut(
        df["profile_quality_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Power"]
    )

    print(f"\n[ProfileQuality] Score distribution:")
    print(df["quality_tier"].value_counts().sort_index().to_string())

    return df


def analyze_engagement_predictors(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """
    Analyze which item features predict higher engagement.
    Equivalent to 'which profile attributes predict conversation starts'.
    """
    print("\n[EngagementAnalysis] Analyzing engagement predictors...")
    merged = ratings.merge(movies[["movieId","genre_count","year"]], on="movieId", how="left")

    results = {}

    # Genre count vs rating
    corr, pval = stats.pearsonr(
        merged["genre_count"].fillna(0),
        merged["rating"]
    )
    results["genre_count_vs_rating"] = {
        "correlation": round(float(corr), 4),
        "p_value":     round(float(pval), 6),
        "significant": bool(pval < 0.05),
        "insight":     f"Genre count has {abs(corr):.3f} correlation with rating "
                       f"({'significant' if pval < 0.05 else 'not significant'})"
    }

    # Rating volume by genre
    merged["genres_list"] = merged["genres"].str.split("|") if "genres" in merged else merged.get("genres_list", "")

    # High vs low raters engagement
    user_avg = ratings.groupby("userId")["rating"].mean()
    high_raters = user_avg[user_avg >= 4.0].index
    low_raters  = user_avg[user_avg < 3.0].index

    high_count = ratings[ratings["userId"].isin(high_raters)]["userId"].value_counts().mean() if len(high_raters) > 0 else 0
    low_count  = ratings[ratings["userId"].isin(low_raters)]["userId"].value_counts().mean()

    results["high_vs_low_rater_volume"] = {
        "high_rater_avg_ratings": round(float(high_count), 1),
        "low_rater_avg_ratings":  round(float(low_count), 1),
        "lift":                   round(float(high_count / (low_count + 1e-9)), 2),
        "insight": f"High-rating users give {high_count:.0f} ratings on average vs "
                   f"{low_count:.0f} for low-rating users — "
                   f"{high_count/low_count:.1f}x more engaged"
    }

    # Year vs rating
    year_data = merged.dropna(subset=["year"])
    if len(year_data) > 100:
        corr_y, pval_y = stats.pearsonr(year_data["year"], year_data["rating"])
        results["recency_vs_rating"] = {
            "correlation": round(float(corr_y), 4),
            "p_value":     round(float(pval_y), 6),
            "insight":     f"Item recency has {abs(corr_y):.3f} correlation with rating"
        }

    for key, val in results.items():
        print(f"  {key}: {val['insight']}")

    return results


if __name__ == "__main__":
    from src.data_loader import load_all
    data = load_all()
    profiles = compute_user_profiles(data["ratings"], data["movies"])
    profiles = score_profile_quality(profiles)
    predictors = analyze_engagement_predictors(data["ratings"], data["movies"])
    print("\nTop 5 Power users:")
    print(profiles[profiles["quality_tier"]=="Power"][
        ["userId","rating_count","genre_entropy","profile_quality_score"]
    ].head(5).to_string(index=False))
