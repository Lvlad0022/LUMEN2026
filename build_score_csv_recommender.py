"""Build a pickle model for the recommendation API from a wide score CSV."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from score_csv_recommender import WideCsvScoreRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a pickled recommender from a wide score CSV.")
    parser.add_argument(
        "--csv-path",
        default=r"C:\Users\lovro\Desktop\hackatoni\LUMEN2026\classic_methods\output_final\final_wide_recommendation_scores.csv",
        help="Path to the wide score CSV.",
    )
    parser.add_argument(
        "--output-model",
        default=r"C:\Users\lovro\Desktop\hackatoni\LUMEN2026\score_csv_recommender.pkl",
        help="Path where the pickled model will be written.",
    )
    parser.add_argument("--customer-col", default="CustomerID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = WideCsvScoreRecommender(args.csv_path, customer_col=args.customer_col)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb") as handle:
        pickle.dump(model, handle)

    print(f"saved model: {output_model}")
    print(f"customers: {len(model.recommendations_by_customer)}")


if __name__ == "__main__":
    main()
