"""
Module 3: Trust Signal Detection
Identifies anomalous user behavior using statistical rules.
No complex model — simple heuristics that work and can be explained clearly.
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrustSignal:
    user_id: int
    signal_type: str
    severity: str       # LOW / MEDIUM / HIGH
    description: str
    metric_value: float
    threshold: float


def detect_rating_spam(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Detect users who rate unusually fast — bulk rating behavior.
    Real users take time between ratings. Bots or spam accounts don't.
    """
    user_ts = ratings.sort_values(["userId","timestamp"])
    user_ts["time_gap_seconds"] = user_ts.groupby("userId")["timestamp"].diff().dt.total_seconds()

    rapid = user_ts[user_ts["time_gap_seconds"] < 2].groupby("userId").size()
    total = user_ts.groupby("userId").size()
    rapid_pct = (rapid / total).fillna(0)

    flagged = rapid_pct[rapid_pct > 0.8].reset_index()
    flagged.columns = ["userId", "rapid_rating_pct"]
    flagged["signal_type"] = "rapid_bulk_rating"
    flagged["severity"]    = flagged["rapid_rating_pct"].apply(
        lambda x: "HIGH" if x > 0.8 else "MEDIUM"
    )
    return flagged


def detect_rating_uniformity(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Detect users who give the same rating to everything.
    Zero variance in ratings = no real engagement or opinion differentiation.
    """
    user_stats = ratings.groupby("userId")["rating"].agg(["std","count"]).reset_index()
    user_stats.columns = ["userId", "rating_std", "rating_count"]

    # Only flag users with enough ratings to be meaningful
    uniform = user_stats[
        (user_stats["rating_std"] < 0.1) &
        (user_stats["rating_count"] >= 10)
    ].copy()
    uniform["signal_type"] = "uniform_ratings"
    uniform["severity"]    = "MEDIUM"
    return uniform[["userId","signal_type","severity","rating_std","rating_count"]]


def detect_outlier_raters(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Detect users whose average rating is a statistical outlier.
    Users rating 3+ standard deviations from the population mean.
    """
    user_avg   = ratings.groupby("userId")["rating"].mean()
    pop_mean   = user_avg.mean()
    pop_std    = user_avg.std()
    z_scores   = (user_avg - pop_mean) / pop_std

    outliers = z_scores[z_scores.abs() > 3].reset_index()
    outliers.columns = ["userId", "z_score"]
    outliers["signal_type"] = "outlier_rater"
    outliers["severity"]    = "LOW"
    outliers["direction"]   = outliers["z_score"].apply(
        lambda z: "inflated" if z > 0 else "deflated"
    )
    return outliers


def detect_impossible_patterns(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Detect users with impossible or implausible behavioral patterns:
    - Rating more items per day than is humanly possible
    - Rating items before the dataset's earliest valid timestamp
    """
    daily_counts = ratings.copy()
    daily_counts["date"] = daily_counts["timestamp"].dt.date
    max_daily = daily_counts.groupby(["userId","date"]).size().groupby("userId").max()

    # 50 ratings in a single day is implausible for genuine engagement
    implausible = max_daily[max_daily > 500].reset_index()
    implausible.columns = ["userId", "max_daily_ratings"]
    implausible["signal_type"] = "implausible_daily_volume"
    implausible["severity"]    = implausible["max_daily_ratings"].apply(
        lambda x: "HIGH" if x > 800 else "MEDIUM"
    )
    return implausible


def run_trust_detection(ratings: pd.DataFrame) -> dict:
    """
    Run all trust signal detectors and produce a unified trust report.
    """
    print("[TrustDetector] Running trust signal detection...")

    spam      = detect_rating_spam(ratings)
    uniform   = detect_rating_uniformity(ratings)
    outliers  = detect_outlier_raters(ratings)
    implaus   = detect_impossible_patterns(ratings)

    total_users = ratings["userId"].nunique()

    all_flagged = set(spam["userId"]) | set(uniform["userId"]) | \
                  set(outliers["userId"]) | set(implaus["userId"])

    high_risk = set(
        spam[spam["severity"]=="HIGH"]["userId"].tolist() +
        implaus[implaus["severity"]=="HIGH"]["userId"].tolist()
    )

    report = {
        "total_users":          int(total_users),
        "flagged_users":        int(len(all_flagged)),
        "flag_rate_pct":        round(len(all_flagged)/total_users*100, 2),
        "high_risk_users":      int(len(high_risk)),
        "detectors": {
            "rapid_bulk_rating":       int(len(spam)),
            "uniform_ratings":         int(len(uniform)),
            "outlier_raters":          int(len(outliers)),
            "implausible_daily_volume":int(len(implaus)),
        },
        "methodology": "Rule-based statistical detection — no ML model required. "
                       "Simple heuristics chosen for interpretability and low false-positive rate.",
        "recommendation": (
            "Remove high-risk users from training data before building recommendation models. "
            "Medium/low severity users can be down-weighted rather than excluded."
        )
    }

    print(f"[TrustDetector] {len(all_flagged)} flagged users out of {total_users} "
          f"({report['flag_rate_pct']}%)")
    print(f"  Rapid raters:      {len(spam)}")
    print(f"  Uniform raters:    {len(uniform)}")
    print(f"  Outlier raters:    {len(outliers)}")
    print(f"  Implausible volume:{len(implaus)}")

    return report, all_flagged, high_risk


if __name__ == "__main__":
    from src.data_loader import load_all
    data   = load_all()
    report, flagged, high_risk = run_trust_detection(data["ratings"])
    print(f"\nHigh risk users: {len(high_risk)}")
    print(f"Safe to use for training: {report['total_users'] - len(high_risk):,} users")
