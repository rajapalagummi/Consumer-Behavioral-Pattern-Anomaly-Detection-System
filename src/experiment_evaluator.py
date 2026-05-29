"""
Module 4: A/B Experiment Evaluation Framework
Simulates and evaluates experiments comparing ranking approaches.
Honest evaluation — states what the experiment proves and what it doesn't.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def simulate_ab_experiment(
    ratings: pd.DataFrame,
    predicted_df: pd.DataFrame,
    genre_df,
    movies: pd.DataFrame,
    n_users: int = 200,
    random_state: int = 42,
) -> Dict:
    """
    Simulate A/B test: collaborative filtering vs content-based ranking.
    Control = content-based | Treatment = collaborative filtering
    Metric = predicted rating for held-out items
    """
    np.random.seed(random_state)

    # Users with enough ratings to split
    qualified = ratings.groupby("userId").filter(lambda x: len(x) >= 20)
    user_pool = qualified["userId"].unique()
    sample_size = min(n_users, len(user_pool))
    sampled_users = np.random.choice(user_pool, size=sample_size, replace=False)

    control_scores   = []
    treatment_scores = []

    from src.recommendation_engine import (
        get_collaborative_recommendations,
        get_content_recommendations,
        build_user_taste_profile
    )

    for user_id in sampled_users:
        user_ratings = ratings[ratings["userId"] == user_id]
        if len(user_ratings) < 10:
            continue

        # Hold out last 20% of ratings by time
        user_ratings_sorted = user_ratings.sort_values("timestamp")
        split_idx = int(len(user_ratings_sorted) * 0.8)
        train_ratings = user_ratings_sorted.iloc[:split_idx]
        test_ratings  = user_ratings_sorted.iloc[split_idx:]

        if len(test_ratings) == 0:
            continue

        test_items = set(test_ratings["movieId"])
        actual_ratings = dict(zip(test_ratings["movieId"], test_ratings["rating"]))

        # Control: content-based score for test items
        taste = build_user_taste_profile(user_id, train_ratings, genre_df)
        if taste.sum() > 0 and len(test_items) > 0:
            available = [m for m in test_items if m in genre_df.index]
            if available:
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(
                    taste.reshape(1, -1),
                    genre_df.loc[available].values
                )[0]
                # Scale similarity to rating range (0-1 → 1-5)
                content_score = 1 + sims.mean() * 4
                actual_score  = np.mean([actual_ratings[m] for m in available])
                control_scores.append(abs(content_score - actual_score))

        # Treatment: collaborative filtering score for test items
        if user_id in predicted_df.index:
            available_collab = [m for m in test_items if m in predicted_df.columns]
            if available_collab:
                collab_preds  = predicted_df.loc[user_id, available_collab].values
                actual_scores = [actual_ratings[m] for m in available_collab]
                mae_collab    = np.mean(np.abs(collab_preds - np.array(actual_scores)))
                treatment_scores.append(mae_collab)

    if len(control_scores) < 20 or len(treatment_scores) < 20:
        return {"error": "Insufficient data for valid experiment", "n_control": len(control_scores)}

    # Statistical test
    t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
    control_mean    = np.mean(control_scores)
    treatment_mean  = np.mean(treatment_scores)
    effect_size     = (control_mean - treatment_mean) / np.std(control_scores + treatment_scores)

    winner = "collaborative" if treatment_mean < control_mean else "content-based"
    significant = p_value < 0.05

    result = {
        "experiment": "Collaborative Filtering vs Content-Based Ranking",
        "metric":     "Mean Absolute Error on held-out ratings (lower is better)",
        "control": {
            "name":   "Content-Based Ranking",
            "n":      len(control_scores),
            "mean_mae": round(float(control_mean), 4),
            "std":    round(float(np.std(control_scores)), 4),
        },
        "treatment": {
            "name":   "Collaborative Filtering (SVD)",
            "n":      len(treatment_scores),
            "mean_mae": round(float(treatment_mean), 4),
            "std":    round(float(np.std(treatment_scores)), 4),
        },
        "statistics": {
            "t_statistic":  round(float(t_stat), 4),
            "p_value":      round(float(p_value), 6),
            "effect_size":  round(float(effect_size), 4),
            "significant":  bool(significant),
        },
        "result": {
            "winner":       winner,
            "improvement":  round(abs(control_mean - treatment_mean) / control_mean * 100, 2),
            "conclusion":   (
                f"{'Statistically significant' if significant else 'Not statistically significant'} "
                f"difference (p={p_value:.4f}). "
                f"{winner.title()} approach has lower prediction error "
                f"({'deploy' if significant else 'inconclusive — run longer experiment'})."
            ),
        },
        "honest_caveats": [
            "Experiment simulated on historical data — real A/B test would measure live engagement",
            "MAE on held-out ratings is a proxy metric — actual click/engagement rate may differ",
            "Cold-start users excluded — results apply only to users with 20+ ratings",
            f"Effect size {abs(effect_size):.3f} — {'practically meaningful' if abs(effect_size) > 0.2 else 'small practical difference'}",
        ]
    }

    print(f"\n[Experiment] A/B Results:")
    print(f"  Control  (content-based): MAE={control_mean:.4f} (n={len(control_scores)})")
    print(f"  Treatment (collaborative): MAE={treatment_mean:.4f} (n={len(treatment_scores)})")
    print(f"  p-value: {p_value:.4f} | Significant: {significant}")
    print(f"  Winner: {winner} | {result['result']['conclusion']}")

    return result


if __name__ == "__main__":
    from src.data_loader import load_all
    from src.recommendation_engine import (
        build_user_item_matrix, train_svd_model, build_item_profiles
    )
    data = load_all()
    matrix = build_user_item_matrix(data["ratings"])
    _, pred_df, _, _ = train_svd_model(matrix)
    genre_df, _ = build_item_profiles(data["movies"])
    result = simulate_ab_experiment(data["ratings"], pred_df, genre_df, data["movies"])
    print(f"\nConclusion: {result['result']['conclusion']}")
