"""Run the selected VAE embedding-distance Katz setup across processed datasets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIC_ROOT = REPO_ROOT / "classic_methods"
SRC_DIR = CLASSIC_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import Pipeline
from validation.offline import (
    EvaluationSplitResult,
    hold_out_last_novel_tail_item,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


DEFAULT_DATASETS = [
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_2.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_3.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_4.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_5.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_6.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the fixed VAE embedding-distance Katz model on each processed LUMEN dataset."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[str(path) for path in DEFAULT_DATASETS],
        help="Processed CSV paths. Defaults to LUMEN_DS_processed through LUMEN_DS_processed_6.",
    )
    parser.add_argument("--search-name", default="final_embedding_distance_katz_datasets")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--users-per-repetition", type=int, default=20)
    parser.add_argument("--recommendation-k", type=int, default=20)
    parser.add_argument("--metrics-k", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=1)
    parser.add_argument("--customer-customer-weight", type=float, default=0.07)
    parser.add_argument("--customer-item-weight", type=float, default=1.0)
    parser.add_argument("--normalize-customer-customer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-customer-item", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument(
        "--katz-max-nnz-per-row",
        type=int,
        default=1000,
        help="Keep only the strongest Katz scores per row after each iteration to avoid sparse matrix blow-up.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write runtime configs; do not run validation.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def dataset_label(path: Path) -> str:
    name = path.stem
    if name == "LUMEN_DS_processed":
        return "processed_1"
    return name.replace("LUMEN_DS_", "")


def write_runtime_configs(
    *,
    dataset_path: Path,
    config_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    embedding_config_path = config_dir / "from_embedding_distance.yaml"
    katz_config_path = config_dir / "katz.yaml"
    pipeline_config_path = config_dir / "pipeline.yaml"
    validation_config_path = config_dir / "validation.yaml"
    base_config_path = config_dir / "base.yaml"
    summary_path = output_dir / "validation_summary.json"
    config_artifact_path = output_dir / "validation_config_bundle.json"

    write_text(
        embedding_config_path,
        f"""
        _target_: models.similarity_matrix.from_embedding_distance.embedding_distance_to_similarity
        method: fit
        params:
          customer_col: CustomerID
          item_col: Item Code
          item_idx_col: item_idx
          metric: euclidean
          similarity_kernel: rbf
          n_neighbors: {int(args.n_neighbors)}
          bandwidth: null
          include_self: true
          symmetrize: max
          customer_customer_weight: {float(args.customer_customer_weight)}
          customer_item_weight: {float(args.customer_item_weight)}
          normalize_customer_customer: {str(bool(args.normalize_customer_customer)).lower()}
          normalize_customer_item: {str(bool(args.normalize_customer_item)).lower()}
          drop_missing_customer_id: true
          drop_missing_item_code: true
        inputs:
          dataframe: dataframe
          embedding: embedding
        """,
    )

    write_text(
        katz_config_path,
        f"""
        _target_: models.similarity_matrix.katz.KatzRecommender
        method: fit
        params:
          beta: {float(args.beta)}
          max_iter: 100
          tol: 1.0e-10
          include_self: false
          symmetric_graph_normalization: true
          max_nnz_per_row: {int(args.katz_max_nnz_per_row)}
        inputs:
          combined_matrix: combined_matrix
          customer_idx: customer_idx
          item_idx: item_idx
        """,
    )

    write_text(
        pipeline_config_path,
        f"""
        data:
          csv_path: {dataset_path}

        output:
          dir: {output_dir}

        save_intermediates: false

        stage1:
          config_path: config/recommendation/data_processing/preprocessing_impute_only.yaml

        stage2:
          config_path: config/recommendation/embeddings/customer_features.yaml

        stage3:
          config_path: config/recommendation/embeddings/vae_embedding.yaml

        stage4:
          config_path: {embedding_config_path}

        stage5:
          config_path: {katz_config_path}
        """,
    )

    write_text(
        validation_config_path,
        f"""
        seed: ${{project.seed}}

        data:
          csv_path: {dataset_path}
          user_col: CustomerID
          item_col: item_idx
          date_col: Order Date

        pipeline:
          recommendation_config_path: {pipeline_config_path}
          recommendation_k: {int(args.recommendation_k)}

        sampling:
          users_per_repetition: {int(args.users_per_repetition)}

        masking:
          strategy: last_novel_tail_item
          tail_fraction: 0.2
          min_training_customer_support: 3

        metrics:
          k: {int(args.metrics_k)}

        output:
          summary_path: {summary_path}
          config_artifact_path: {config_artifact_path}

        mlflow:
          enabled: false
          tracking_uri: null
          experiment_name: final_embedding_distance_katz_datasets
          run_name: {dataset_label(dataset_path)}
          tags:
            dataset: {dataset_path.name}
        """,
    )

    write_text(
        base_config_path,
        f"""
        project:
          name: classic_methods
          seed: 42

        validation_bool: true
        validation_repetitions: {int(args.repetitions)}

        paths:
          config_path: config/paths.yaml

        pipeline:
          config_path: {pipeline_config_path}

        validation:
          config_path: {validation_config_path}
        """,
    )
    return base_config_path


def flatten_summary(dataset_path: Path, base_config_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    row: dict[str, Any] = {
        "dataset": dataset_path.name,
        "base_config_path": str(base_config_path),
        "summary_path": str(base_config_path.parent.parent.parent / "unused"),
        "repetitions_run": int(summary.get("repetitions_run", 0)),
    }
    for metric_name in ("ndcg_at_k", "recall_at_k", "mrr_at_k"):
        stats = dict(metrics.get(metric_name, {}) or {})
        row[f"{metric_name}_mean"] = float(stats.get("mean", 0.0))
        row[f"{metric_name}_std"] = float(stats.get("std", 0.0))
    return row


def write_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    csv_path = output_root / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_root / "results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_split(dataset_path: Path) -> EvaluationSplitResult:
    dataframe = read_dataset(dataset_path)
    return hold_out_last_novel_tail_item(
        dataframe,
        user_col="CustomerID",
        item_col="item_idx",
        date_col="Order Date",
        tail_fraction=0.2,
        min_training_customer_support=3,
    )


def build_cohorts(
    dataset_paths: list[Path],
    splits: dict[Path, EvaluationSplitResult],
    output_root: Path,
) -> dict[str, set[int]]:
    cohorts: dict[str, set[int]] = {}
    seen_users: set[int] = set()
    definition_rows: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        label = dataset_label(dataset_path)
        validation_users = {int(user_id) for user_id in splits[dataset_path].evaluated_user_ids}
        new_users = validation_users - seen_users
        cohorts[label] = new_users
        seen_users.update(new_users)
        definition_rows.append(
            {
                "cohort": label,
                "source_dataset": dataset_path.name,
                "cohort_size": len(new_users),
                "cumulative_validation_users": len(seen_users),
                "validation_users_in_dataset": len(validation_users),
            }
        )

    csv_path = output_root / "cohort_definitions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(definition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(definition_rows)

    json_path = output_root / "cohort_users.json"
    json_path.write_text(
        json.dumps(
            {name: sorted(users) for name, users in cohorts.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    return cohorts


def summarize_metric(value: float) -> dict[str, float]:
    return {"mean": float(value), "std": 0.0}


def run_validation_with_cohorts(
    *,
    dataset_path: Path,
    split: EvaluationSplitResult,
    pipeline_config_path: Path,
    base_config_path: Path,
    output_dir: Path,
    output_root: Path,
    cohorts: dict[str, set[int]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    print(
        "[validation] deterministic split: "
        f"evaluating {len(split.evaluated_user_ids)} user(s), "
        f"training rows={len(split.masked_df)}"
    )

    recommendation_pipeline = Pipeline(pipeline_config_path, show_progress=True)
    artifacts = recommendation_pipeline.run(
        initial_artifacts={"dataframe": split.masked_df},
        save_artifacts=False,
        return_recommendations=True,
        recommendation_k=int(args.recommendation_k),
    )
    predictions = artifacts["recommendations"].value
    evaluated_ground_truth = {
        int(user_id): split.ground_truth[user_id]
        for user_id in split.evaluated_user_ids
        if int(user_id) in predictions
    }

    overall_ndcg = ndcg_at_k(predictions, evaluated_ground_truth, k=int(args.metrics_k))
    overall_recall = recall_at_k(predictions, evaluated_ground_truth, k=int(args.metrics_k))
    overall_mrr = mrr_at_k(predictions, evaluated_ground_truth, k=int(args.metrics_k))
    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "requested_repetitions": int(args.repetitions),
        "repetitions_run": 1,
        "seed": 42,
        "source_csv_path": str(dataset_path),
        "masking_strategy": "last_novel_tail_item",
        "recommendation_k": int(args.recommendation_k),
        "metrics_k": int(args.metrics_k),
        "metrics": {
            "ndcg_at_k": summarize_metric(overall_ndcg),
            "recall_at_k": summarize_metric(overall_recall),
            "mrr_at_k": summarize_metric(overall_mrr),
        },
        "split_details": json_safe(dict(split.split_details or {})),
        "repetition_results": [
            {
                "repetition": 1,
                "sampled_user_count": int(len(split.evaluated_user_ids)),
                "masked_user_count": int(len(split.evaluated_user_ids)),
                "evaluated_user_count": int(len(evaluated_ground_truth)),
                "skipped_user_ids": [int(user_id) for user_id in split.skipped_user_ids],
                "users_missing_predictions": [
                    int(user_id)
                    for user_id in split.evaluated_user_ids
                    if int(user_id) not in predictions
                ],
                "ndcg_at_k": float(overall_ndcg),
                "recall_at_k": float(overall_recall),
                "mrr_at_k": float(overall_mrr),
            }
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_bundle_path = output_dir / "validation_config_bundle.json"
    config_bundle_path.write_text(
        json.dumps(
            {
                "base_config_path": str(base_config_path),
                "recommendation_pipeline_config_path": str(pipeline_config_path),
                "summary_path": str(summary_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cohort_rows: list[dict[str, Any]] = []
    for cohort_name, cohort_users in cohorts.items():
        cohort_ground_truth = {
            user_id: target
            for user_id, target in evaluated_ground_truth.items()
            if user_id in cohort_users
        }
        evaluable_count = len(cohort_ground_truth)
        hits = 0
        if evaluable_count:
            for user_id, target_item in cohort_ground_truth.items():
                if target_item in predictions.get(user_id, [])[: int(args.metrics_k)]:
                    hits += 1
            cohort_recall = hits / evaluable_count
        else:
            cohort_recall = None
        cohort_rows.append(
            {
                "trained_dataset": dataset_path.name,
                "trained_dataset_label": dataset_label(dataset_path),
                "cohort": cohort_name,
                "cohort_size": len(cohort_users),
                "evaluable_count": evaluable_count,
                "hits_at_k": hits,
                "k": int(args.metrics_k),
                "recall_at_k": cohort_recall,
            }
        )

    write_cohort_recall_rows(cohort_rows, output_root)
    return summary, summary["metrics"], cohort_rows


def write_cohort_recall_rows(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    path = output_root / "cohort_recall.csv"
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    session_code = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_root = CLASSIC_ROOT / "config" / "final_runtime" / args.search_name / session_code
    output_root = CLASSIC_ROOT / "output_final" / args.search_name / session_code
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    dataset_paths = [Path(path).resolve() for path in args.datasets]
    splits: dict[Path, EvaluationSplitResult] = {}
    for index, dataset_path in enumerate(dataset_paths, start=1):
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
        if not args.dry_run:
            print(f"[final-katz] building validation split for {dataset_path.name}")
            splits[dataset_path] = build_split(dataset_path)

    cohorts: dict[str, set[int]] = {}
    if not args.dry_run:
        cohorts = build_cohorts(dataset_paths, splits, output_root)
        print(f"[final-katz] cohort definitions saved to {output_root / 'cohort_definitions.csv'}")

    for index, dataset_path in enumerate(dataset_paths, start=1):
        label = dataset_label(dataset_path)
        config_dir = config_root / f"{index:02d}_{label}"
        output_dir = output_root / f"{index:02d}_{label}"
        base_config_path = write_runtime_configs(
            dataset_path=dataset_path,
            config_dir=config_dir,
            output_dir=output_dir,
            args=args,
        )
        print(f"[final-katz] prepared {dataset_path.name}: {base_config_path}")
        if args.dry_run:
            continue

        summary, _, _ = run_validation_with_cohorts(
            dataset_path=dataset_path,
            split=splits[dataset_path],
            pipeline_config_path=config_dir / "pipeline.yaml",
            base_config_path=base_config_path,
            output_dir=output_dir,
            output_root=output_root,
            cohorts=cohorts,
            args=args,
        )
        row = flatten_summary(dataset_path, base_config_path, summary)
        row["summary_path"] = str(output_dir / "validation_summary.json")
        rows.append(row)
        write_results(rows, output_root)
        print(
            f"[final-katz] {dataset_path.name}: "
            f"ndcg={row['ndcg_at_k_mean']:.4f}, "
            f"recall={row['recall_at_k_mean']:.4f}, "
            f"mrr={row['mrr_at_k_mean']:.4f}"
        )

    print(f"[final-katz] configs: {config_root}")
    print(f"[final-katz] outputs: {output_root}")


if __name__ == "__main__":
    main()
