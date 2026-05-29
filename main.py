"""
Behavioral Signal Engine — Main Orchestrator
MovieLens real data | User engagement profiling | Recommendation ranking
Trust signal detection | A/B experiment evaluation

Usage:
    python3 main.py                          # Full pipeline
    python3 main.py --data path/to/data/     # Custom data directory
    python3 main.py --skip-experiment        # Skip A/B simulation (faster)
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "outputs"

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         Behavioral Signal Engine                                 ║
║         User Engagement Profiling & Recommendation Intelligence  ║
║                                                                  ║
║  Module 1: User Engagement Profiling & Profile Quality Scoring   ║
║  Module 2: Recommendation Ranking (Collaborative vs Content)     ║
║  Module 3: Trust Signal Detection (Anomaly & Spam Detection)     ║
║  Module 4: A/B Experiment Evaluation Framework                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def generate_dashboard(profiles, comparison, trust_report,
                        experiment_result, ratings, movies):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Behavioral Signal Engine — Analytics Dashboard\nMovieLens Real Data",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    BLUE   = "#2E75B6"
    RED    = "#C93828"
    GREEN  = "#0A8F5C"
    ORANGE = "#B87200"
    GRAY   = "#595959"

    # ── Panel 1: Profile Quality Distribution ────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    tier_counts = profiles["quality_tier"].value_counts().sort_index()
    colors_tier = [GRAY, BLUE, GREEN, ORANGE]
    ax1.bar(tier_counts.index, tier_counts.values,
            color=colors_tier[:len(tier_counts)], edgecolor="white", alpha=0.85)
    ax1.set_title("User Profile Quality Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Quality Tier"); ax1.set_ylabel("User Count")
    ax1.spines[["top","right"]].set_visible(False)
    for i, (tier, count) in enumerate(tier_counts.items()):
        ax1.text(i, count + 1, str(count), ha="center", fontsize=9, fontweight="bold")

    # ── Panel 2: Engagement Signal Correlation ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    signal_cols = ["rating_count", "genre_entropy", "activity_consistency",
                   "rating_discrimination", "profile_quality_score"]
    available = [c for c in signal_cols if c in profiles.columns]
    if len(available) >= 3:
        corr_matrix = profiles[available].corr()
        sns.heatmap(corr_matrix, ax=ax2, cmap="RdBu_r", center=0,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    xticklabels=[c.replace("_"," ")[:10] for c in available],
                    yticklabels=[c.replace("_"," ")[:10] for c in available])
    ax2.set_title("Engagement Signal Correlations", fontsize=11, fontweight="bold")

    # ── Panel 3: Trust Signal Summary ───────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if "detectors" in trust_report:
        det = trust_report["detectors"]
        labels = [k.replace("_", "\n") for k in det.keys()]
        values = list(det.values())
        bar_colors = [RED if v > 10 else ORANGE if v > 5 else GREEN for v in values]
        ax3.barh(labels, values, color=bar_colors, edgecolor="white", alpha=0.85)
        ax3.set_title(f"Trust Signals Detected\n({trust_report.get('flagged_users',0)} flagged / "
                      f"{trust_report.get('total_users',0)} total)",
                      fontsize=11, fontweight="bold")
        ax3.set_xlabel("Users Flagged")
        ax3.spines[["top","right"]].set_visible(False)

    # ── Panel 4: Profile Quality Score Distribution ──────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.hist(profiles["profile_quality_score"], bins=40,
             color=BLUE, edgecolor="white", alpha=0.85)
    ax4.axvline(profiles["profile_quality_score"].mean(),
                color=RED, linestyle="--", lw=2,
                label=f"Mean: {profiles['profile_quality_score'].mean():.3f}")
    ax4.axvline(0.25, color=GRAY, linestyle=":", lw=1.5, alpha=0.7, label="Tier boundaries")
    ax4.axvline(0.50, color=GRAY, linestyle=":", lw=1.5, alpha=0.7)
    ax4.axvline(0.75, color=GRAY, linestyle=":", lw=1.5, alpha=0.7)
    ax4.set_title("Profile Quality Score Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Profile Quality Score (0-1)")
    ax4.set_ylabel("User Count")
    ax4.legend(fontsize=9); ax4.spines[["top","right"]].set_visible(False)

    # ── Panel 5: Rating volume distribution ─────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    user_counts = ratings.groupby("userId").size()
    ax5.hist(user_counts, bins=30, color=GREEN, edgecolor="white", alpha=0.85)
    ax5.axvline(user_counts.mean(), color=RED, linestyle="--", lw=2,
                label=f"Mean: {user_counts.mean():.0f}")
    ax5.axvline(user_counts.median(), color=ORANGE, linestyle="--", lw=2,
                label=f"Median: {user_counts.median():.0f}")
    ax5.set_title("Ratings per User Distribution", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Number of Ratings"); ax5.set_ylabel("User Count")
    ax5.legend(fontsize=9); ax5.spines[["top","right"]].set_visible(False)

    # ── Panel 6: A/B Experiment Results ─────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")

    if "error" not in experiment_result and "control" in experiment_result:
        ctrl  = experiment_result["control"]
        treat = experiment_result["treatment"]
        stats_r = experiment_result["statistics"]
        res   = experiment_result["result"]

        summary_text = (
            f"A/B EXPERIMENT: {experiment_result['experiment']}\n\n"
            f"Control  ({ctrl['name']}):   MAE = {ctrl['mean_mae']:.4f}  (n={ctrl['n']:,})\n"
            f"Treatment ({treat['name']}): MAE = {treat['mean_mae']:.4f}  (n={treat['n']:,})\n\n"
            f"p-value: {stats_r['p_value']:.4f}  |  "
            f"Effect size: {stats_r['effect_size']:.4f}  |  "
            f"Significant: {stats_r['significant']}\n\n"
            f"CONCLUSION: {res['conclusion']}\n\n"
            f"CAVEATS: {experiment_result['honest_caveats'][0]}"
        )
        ax6.text(0.05, 0.9, summary_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#E6EAF0", alpha=0.8))
    else:
        ax6.text(0.5, 0.5, "A/B Experiment skipped or insufficient data",
                 ha="center", va="center", fontsize=12, color=GRAY,
                 transform=ax6.transAxes)

    fig.text(0.5, 0.01,
             f"Data: MovieLens | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | rajapalagummi.com",
             ha="center", fontsize=8, color="#999", style="italic")

    path = os.path.join(OUTPUT_DIR, "behavioral_signal_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Dashboard] → {path}")
    return path


def run_pipeline(data_dir: str = "data", skip_experiment: bool = False):
    print(BANNER)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    print("=" * 60)
    print("  Loading MovieLens Dataset")
    print("=" * 60)

    from src.data_loader import load_all
    import src.data_loader as dl
    dl.DATA_DIR = data_dir
    data = load_all()

    ratings = data["ratings"]
    movies  = data["movies"]

    # ── Module 1: Engagement Profiling ───────────────────────────
    print("\n" + "=" * 60)
    print("  MODULE 1: User Engagement Profiling & Profile Quality")
    print("=" * 60)

    from src.engagement_profiler import (
        compute_user_profiles, score_profile_quality,
        analyze_engagement_predictors
    )
    profiles    = compute_user_profiles(ratings, movies)
    profiles    = score_profile_quality(profiles)
    predictors  = analyze_engagement_predictors(ratings, movies)

    # ── Module 2: Recommendation Ranking ────────────────────────
    print("\n" + "=" * 60)
    print("  MODULE 2: Recommendation Ranking")
    print("=" * 60)

    from src.recommendation_engine import (
        build_user_item_matrix, train_svd_model,
        evaluate_collaborative, build_item_profiles,
        compare_approaches, get_collaborative_recommendations,
        get_content_recommendations
    )
    matrix              = build_user_item_matrix(ratings)
    svd, pred_df, U, Vt = train_svd_model(matrix)
    collab_metrics      = evaluate_collaborative(ratings, matrix, pred_df)
    genre_df, mlb       = build_item_profiles(movies)
    comparison          = compare_approaches(ratings, movies, collab_metrics)

    # ── Module 3: Trust Signal Detection ────────────────────────
    print("\n" + "=" * 60)
    print("  MODULE 3: Trust Signal Detection")
    print("=" * 60)

    from src.trust_detector import run_trust_detection
    trust_report, flagged_users, high_risk = run_trust_detection(ratings)

    # ── Module 4: A/B Experiment ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODULE 4: A/B Experiment Evaluation")
    print("=" * 60)

    experiment_result = {}
    if not skip_experiment:
        from src.experiment_evaluator import simulate_ab_experiment
        experiment_result = simulate_ab_experiment(
            ratings, pred_df, genre_df, movies, n_users=200
        )
    else:
        print("[Experiment] Skipped (--skip-experiment flag)")

    # ── Generate dashboard ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating Dashboard & Reports")
    print("=" * 60)

    generate_dashboard(profiles, comparison, trust_report,
                       experiment_result, ratings, movies)

    # Save summary JSON
    summary = {
        "generated_at":   datetime.now().isoformat(),
        "dataset":        "MovieLens ml-latest-small",
        "users":          int(ratings["userId"].nunique()),
        "items":          int(ratings["movieId"].nunique()),
        "ratings":        int(len(ratings)),
        "profile_quality":{
            "power_users":  int((profiles["quality_tier"] == "Power").sum()),
            "high_users":   int((profiles["quality_tier"] == "High").sum()),
            "medium_users": int((profiles["quality_tier"] == "Medium").sum()),
            "low_users":    int((profiles["quality_tier"] == "Low").sum()),
            "avg_score":    round(float(profiles["profile_quality_score"].mean()), 4),
        },
        "recommendation": {
            "collaborative_rmse": collab_metrics.get("rmse"),
            "collaborative_mae":  collab_metrics.get("mae"),
            "matrix_shape":       list(matrix.shape),
        },
        "trust": {
            "flagged_users":   int(len(flagged_users)),
            "high_risk_users": int(len(high_risk)),
            "flag_rate_pct":   trust_report.get("flag_rate_pct"),
        },
        "experiment": experiment_result if "error" not in experiment_result else {},
        "engagement_predictors": predictors,
    }

    summary_path = os.path.join(OUTPUT_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"""
{'='*60}
  ✓ PIPELINE COMPLETE
{'='*60}

📊 Key Results:
   Users:                {summary['users']:,}
   Items:                {summary['items']:,}
   Ratings:              {summary['ratings']:,}

   Profile Quality:
   Power users:          {summary['profile_quality']['power_users']:,}
   Avg quality score:    {summary['profile_quality']['avg_score']:.4f}

   Recommendation:
   Collaborative RMSE:   {summary['recommendation']['collaborative_rmse']}
   Collaborative MAE:    {summary['recommendation']['collaborative_mae']}

   Trust:
   Flagged users:        {summary['trust']['flagged_users']:,} ({summary['trust']['flag_rate_pct']}%)
   High risk:            {summary['trust']['high_risk_users']:,}

📁 Outputs:
   outputs/behavioral_signal_dashboard.png
   outputs/pipeline_summary.json
""")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Signal Engine")
    parser.add_argument("--data",            type=str, default="data",
                        help="Path to directory containing MovieLens CSV files")
    parser.add_argument("--skip-experiment", action="store_true",
                        help="Skip A/B experiment simulation (faster run)")
    args = parser.parse_args()

    run_pipeline(data_dir=args.data, skip_experiment=args.skip_experiment)
