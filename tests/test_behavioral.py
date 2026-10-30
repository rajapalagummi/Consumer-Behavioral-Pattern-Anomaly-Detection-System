import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trust_detector import detect_rating_spam, detect_rating_uniformity, detect_outlier_raters, detect_impossible_patterns, run_trust_detection
from src.recommendation_engine import build_user_item_matrix, train_svd_model


def ratings_frame(rows):
    df = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def normal_users(n_users=20, n_items=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        for i in range(1, n_items + 1):
            rows.append((u, i, float(rng.choice([2.0, 3.0, 3.5, 4.0, 5.0])), 1_000_000 + u * 100_000 + i * 3600))
    return rows


def test_rapid_bulk_rater_flagged():
    rows = normal_users() + [(999, i, 4.0, 2_000_000 + i) for i in range(1, 21)]
    flagged = detect_rating_spam(ratings_frame(rows))
    assert 999 in set(flagged["userId"])
    assert 1 not in set(flagged["userId"])


def test_uniform_rater_flagged():
    rows = normal_users() + [(999, i, 3.0, 2_000_000 + i * 3600) for i in range(1, 16)]
    flagged = detect_rating_uniformity(ratings_frame(rows))
    assert set(flagged["userId"]) == {999}


def test_outlier_rater_flagged():
    rng = np.random.default_rng(1)
    rows = [(u, i, float(rng.choice([3.0, 3.5, 4.0])), 1_000_000 + u * 100_000 + i * 3600) for u in range(1, 60) for i in range(1, 11)]
    rows += [(999, i, 0.5, 9_000_000 + i * 3600) for i in range(1, 11)]
    flagged = detect_outlier_raters(ratings_frame(rows))
    assert 999 in set(flagged["userId"])
    assert flagged.loc[flagged["userId"] == 999, "direction"].iloc[0] == "deflated"


def test_implausible_daily_volume_flagged():
    rows = normal_users() + [(999, i, 4.0, 3_000_000 + i * 10) for i in range(1, 601)]
    flagged = detect_impossible_patterns(ratings_frame(rows))
    assert set(flagged["userId"]) == {999}


def test_trust_report_counts():
    rows = normal_users() + [(999, i, 3.0, 2_000_000 + i) for i in range(1, 21)]
    report, flagged, high_risk = run_trust_detection(ratings_frame(rows))
    assert report["total_users"] == 21
    assert 999 in flagged
    assert 0 <= report["flag_rate_pct"] <= 100


def test_matrix_filters_sparse_users_and_items():
    rows = normal_users(n_users=10, n_items=10) + [(999, 1, 4.0, 5_000_000), (1, 999, 4.0, 5_000_001)]
    matrix = build_user_item_matrix(ratings_frame(rows), min_ratings=5)
    assert 999 not in matrix.index
    assert 999 not in matrix.columns
    assert matrix.shape == (10, 10)


def test_svd_predictions_in_rating_range():
    matrix = build_user_item_matrix(ratings_frame(normal_users()), min_ratings=5)
    svd, pred, U, Vt = train_svd_model(matrix, n_components=5)
    assert pred.shape == matrix.shape
    assert pred.values.min() >= 0.5 and pred.values.max() <= 5.0
