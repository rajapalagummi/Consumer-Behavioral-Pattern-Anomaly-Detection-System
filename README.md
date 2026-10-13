# Behavioral Signal Engine
## User Engagement Profiling, Recommendation Ranking & Trust Detection on Real MovieLens Data

---

## Overview

Recommendation systems fail in predictable ways. Collaborative filtering breaks for new users. Content-based filtering breaks for new items. Neither approach handles fake or low-quality behavioral signals gracefully. And most implementations skip the honest step of comparing approaches against each other with a real experiment.

This project builds a complete behavioral signal analysis pipeline on real MovieLens data — 100,836 ratings from 610 users across 9,742 items. It profiles user engagement quality, compares collaborative and content-based ranking approaches, detects anomalous behavioral signals, and evaluates the comparison with a structured A/B framework that explicitly states what the experiment proves and what it doesn't.

---

## What It Builds

### Module 1 — User Engagement Profiling & Profile Quality Scoring

Computes a behavioral profile for each user from their rating history. The profile quality score combines four signals:

- **Volume score** — how many ratings the user has given (depth of engagement)
- **Genre entropy** — Shannon entropy of genre distribution (breadth of interest)
- **Activity consistency** — regularity of engagement over time using coefficient of variation
- **Rating discrimination** — whether the user differentiates between good and bad items, or rates everything the same

These four signals are normalized and combined with fixed weights into a single profile quality score between 0 and 1. Users are classified into four tiers: Low, Medium, High, and Power.

The insight this produces is directly applicable to any discovery product: high-quality profiles generate more reliable recommendation signal. Low-quality profiles should be down-weighted, not excluded entirely, because even sparse engagement is signal.

### Module 2 — Recommendation Ranking: Collaborative vs Content-Based

**Collaborative filtering** uses SVD matrix factorization on the user-item rating matrix. It learns 50 latent dimensions that capture patterns in how users rate items relative to each other. The matrix is 98.3% sparse on this dataset — a realistic condition for any early-stage product.

**Content-based ranking** builds a genre vector for each item using multi-label binarization, then constructs a weighted taste profile for each user as the average of their rated items' genre vectors, weighted by rating. Recommendations are ranked by cosine similarity between the user's taste profile and candidate items.

Both approaches are evaluated on held-out ratings. The comparison includes an explicit tradeoff analysis: collaborative filtering cannot handle cold start, content-based cannot handle taste evolution beyond established preferences. The honest recommendation is a hybrid — content-based for new users, collaborative as signal accumulates.

### Module 3 — Trust Signal Detection

Four statistical detectors run on the rating stream:

- **Rapid bulk rating** — users who rate multiple items within 10 seconds of each other, flagging automated or inauthentic behavior
- **Rating uniformity** — users who give identical or near-identical ratings to everything, indicating no genuine engagement differentiation
- **Outlier raters** — users whose average rating is more than 3 standard deviations from the population mean
- **Implausible daily volume** — users who rate more than 50 items in a single day

No model is used. These are explicit statistical rules chosen for interpretability and low false-positive rate. The methodology section of the output explicitly states this and explains why simple heuristics are appropriate here.

### Module 4 — A/B Experiment Evaluation Framework

Simulates a controlled experiment comparing collaborative filtering against content-based ranking. For each user, ratings are split 80/20 by time. The control condition uses content-based scoring on held-out items. The treatment condition uses collaborative filtering predictions.

The evaluation reports:
- Mean absolute error for both conditions
- t-test p-value and effect size
- Whether the difference is statistically significant
- Four explicit caveats about what the experiment does and does not prove

The caveats are not disclaimers — they are the analytical output. Knowing that MAE on held-out historical ratings is a proxy for live engagement metrics is information a product team needs before making a deployment decision.

---

## Key Results on Real MovieLens Data

Running on your real files will produce results similar to:

- **610 users** profiled with engagement quality scores
- **~165 ratings per user** average — dense enough for reliable collaborative filtering
- **Collaborative filtering RMSE ~1.79** — standard for SVD on this dataset
- **Trust detection** flags users with anomalous patterns for review
- **A/B framework** produces statistically testable comparison between approaches

---

## How to Run

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Place MovieLens files in data/ directory
# Download from: https://grouplens.org/datasets/movielens/latest/
# Required: ratings.csv, movies.csv, tags.csv

# 3. Run full pipeline
python3 main.py

# 4. Skip A/B experiment for faster run
python3 main.py --skip-experiment

# 5. Custom data directory
python3 main.py --data /path/to/movielens/files/
```

---

## Generated Outputs

```
outputs/
├── behavioral_signal_dashboard.png   ← Hero image
└── pipeline_summary.json             ← Full results with all metrics
```

---

*Built by Raja Palagummi | rajapalagummi.com | github.com/rajapalagummi*
