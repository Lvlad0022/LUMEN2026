"""Export one wide recommendation-score CSV from cohort-specific Katz models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from validation.offline import hold_out_last_novel_tail_item


DEFAULT_PROCESSED_DATASETS = [
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_2.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_3.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_4.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_5.csv"),
    Path(r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS_processed_6.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train full-data cohort models and export one wide CSV: "
            "CustomerID plus original Item Code score columns."
        )
    )
    parser.add_argument(
        "--processed-datasets",
        nargs="*",
        default=[str(path) for path in DEFAULT_PROCESSED_DATASETS],
    )
    parser.add_argument("--raw-dataset", default=r"C:\Users\lovro\Desktop\hackatoni\LUMEN_DS.csv")
    parser.add_argument(
        "--output-csv",
        default=r"C:\Users\lovro\Desktop\hackatoni\LUMEN2026\classic_methods\output_final\final_wide_recommendation_scores.csv",
    )
    parser.add_argument("--run-name", default="final_wide_recommendation_scores")
    parser.add_argument("--top-k-per-row", type=int, default=1000)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--min-training-customer-support", type=int, default=3)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--customer-customer-weight", type=float, default=0.07)
    parser.add_argument("--customer-item-weight", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--katz-max-nnz-per-row", type=int, default=1000)
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def dataset_label(path: Path) -> str:
    if path.stem == "LUMEN_DS_processed":
        return "processed_1"
    return path.stem.replace("LUMEN_DS_", "")


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_cohorts(
    dataset_paths: list[Path],
    *,
    tail_fraction: float,
    min_training_customer_support: int,
    output_dir: Path,
) -> dict[str, set[int]]:
    cohorts: dict[str, set[int]] = {}
    seen_users: set[int] = set()
    definition_rows: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        dataframe = read_dataset(dataset_path)
        split = hold_out_last_novel_tail_item(
            dataframe,
            user_col="CustomerID",
            item_col="item_idx",
            date_col="Order Date",
            tail_fraction=tail_fraction,
            min_training_customer_support=min_training_customer_support,
        )
        label = dataset_label(dataset_path)
        validation_users = {int(user_id) for user_id in split.evaluated_user_ids}
        cohort_users = validation_users - seen_users
        cohorts[label] = cohort_users
        seen_users.update(cohort_users)
        definition_rows.append(
            {
                "cohort": label,
                "source_dataset": dataset_path.name,
                "cohort_size": len(cohort_users),
                "validation_users_in_dataset": len(validation_users),
                "cumulative_users": len(seen_users),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cohort_definitions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(definition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(definition_rows)
    (output_dir / "cohort_users.json").write_text(
        json.dumps({name: sorted(users) for name, users in cohorts.items()}, indent=2),
        encoding="utf-8",
    )
    return cohorts


def write_pipeline_configs(
    *,
    config_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    args: argparse.Namespace,
    raw_mode: bool,
) -> Path:
    preprocessing_config_path = config_dir / "preprocessing.yaml"
    embedding_config_path = config_dir / "from_embedding_distance.yaml"
    katz_config_path = config_dir / "katz.yaml"
    pipeline_config_path = config_dir / "pipeline.yaml"

    if raw_mode:
        write_text(
            preprocessing_config_path,
            """
            _target_: data_processing.preprocessing.Preprocessor
            method: fit_transform
            params:
              map_item_codes: true
              impute: false
              drop_price_gt_tx: false
              drop_missing_customer_id: true
              drop_missing_item_code: true
              reindex_items_at_end: true
              min_num_purchases_per_customer: null
              min_num_items_per_customer: null
              min_num_purchases_per_item_rows: null
              min_num_unique_customers_per_item: null
              null_rules: []
            inputs:
              df: dataframe
            """,
        )
    else:
        preprocessing_config_path = Path("config/recommendation/data_processing/preprocessing_impute_only.yaml")

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
          normalize_customer_customer: true
          normalize_customer_item: true
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
          config_path: {preprocessing_config_path}

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
    return pipeline_config_path


def row_scores_for_customer(
    *,
    model: Any,
    idx2item: dict[Any, Any],
    customer_id: int,
    top_k: int,
) -> dict[str, float]:
    customer_pos = model.customer_lookup_[int(customer_id)]
    sparse_row = model.item_scores_.getrow(customer_pos).tocsr()
    indices = sparse_row.indices
    data = sparse_row.data
    if data.size == 0:
        return {}

    seen_positions = model._seen_items_by_customer.get(int(customer_id), set())
    keep_mask = np.asarray([int(pos) not in seen_positions for pos in indices], dtype=bool)
    indices = indices[keep_mask]
    data = data[keep_mask]
    if data.size == 0:
        return {}

    if top_k > 0 and data.size > top_k:
        keep = np.argpartition(data, -top_k)[-top_k:]
        indices = indices[keep]
        data = data[keep]

    scores: dict[str, float] = {}
    for position, score in zip(indices.tolist(), data.tolist()):
        item_idx = int(model.item_idx_[int(position)])
        item_code = str(idx2item.get(item_idx, item_idx))
        if np.isfinite(score) and float(score) != 0.0:
            scores[item_code] = float(score)
    return scores


def append_model_rows_to_jsonl(
    *,
    pipeline_config_path: Path,
    cohort_users: set[int],
    row_jsonl_path: Path,
    item_codes: set[str],
    top_k: int,
) -> int:
    pipeline = Pipeline(pipeline_config_path, show_progress=True)
    artifacts = pipeline.run(save_artifacts=False, return_recommendations=False)
    model = artifacts["model"].value
    idx2item = artifacts["idx2item"].value
    item_codes.update(str(item_code) for item_code in idx2item.values())

    exported_count = 0
    with row_jsonl_path.open("a", encoding="utf-8") as handle:
        for customer_id in sorted(cohort_users):
            if int(customer_id) not in model.customer_lookup_:
                continue
            scores = row_scores_for_customer(
                model=model,
                idx2item=idx2item,
                customer_id=int(customer_id),
                top_k=top_k,
            )
            item_codes.update(scores)
            handle.write(json.dumps({"CustomerID": int(customer_id), "scores": scores}) + "\n")
            exported_count += 1
    return exported_count


def write_wide_csv(row_jsonl_path: Path, output_csv: Path, item_codes: set[str]) -> None:
    columns = ["CustomerID"] + sorted(item_codes)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=columns, restval=0.0, extrasaction="ignore")
        writer.writeheader()
        with row_jsonl_path.open("r", encoding="utf-8") as jsonl_handle:
            for line in jsonl_handle:
                record = json.loads(line)
                row = {"CustomerID": record["CustomerID"]}
                row.update(record["scores"])
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    processed_paths = [Path(path).resolve() for path in args.processed_datasets]
    raw_path = Path(args.raw_dataset).resolve()
    output_csv = Path(args.output_csv).resolve()

    session_code = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_root = CLASSIC_ROOT / "config" / "final_runtime" / args.run_name / session_code
    output_root = CLASSIC_ROOT / "output_final" / args.run_name / session_code
    runtime_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for path in [*processed_paths, raw_path]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset does not exist: {path}")

    print("[final-export] building cohorts from processed datasets")
    cohorts = build_cohorts(
        processed_paths,
        tail_fraction=float(args.tail_fraction),
        min_training_customer_support=int(args.min_training_customer_support),
        output_dir=output_root,
    )

    row_jsonl = Path(NamedTemporaryFile(delete=False, suffix=".jsonl").name)
    item_codes: set[str] = set()
    covered_customers: set[int] = set()

    for index, dataset_path in enumerate(processed_paths, start=1):
        label = dataset_label(dataset_path)
        config_dir = runtime_root / f"{index:02d}_{label}"
        model_output_dir = output_root / f"{index:02d}_{label}"
        pipeline_config_path = write_pipeline_configs(
            config_dir=config_dir,
            output_dir=model_output_dir,
            dataset_path=dataset_path,
            args=args,
            raw_mode=False,
        )
        cohort_users = cohorts.get(label, set())
        print(f"[final-export] training {label} on {dataset_path.name}; exporting {len(cohort_users)} cohort users")
        exported = append_model_rows_to_jsonl(
            pipeline_config_path=pipeline_config_path,
            cohort_users=cohort_users,
            row_jsonl_path=row_jsonl,
            item_codes=item_codes,
            top_k=int(args.top_k_per_row),
        )
        covered_customers.update(cohort_users)
        print(f"[final-export] {label}: exported_rows={exported}")

    print("[final-export] training raw fallback model for all remaining customers")
    raw_config_dir = runtime_root / "07_raw_fallback"
    raw_output_dir = output_root / "07_raw_fallback"
    raw_pipeline_config_path = write_pipeline_configs(
        config_dir=raw_config_dir,
        output_dir=raw_output_dir,
        dataset_path=raw_path,
        args=args,
        raw_mode=True,
    )
    raw_pipeline = Pipeline(raw_pipeline_config_path, show_progress=True)
    raw_artifacts = raw_pipeline.run(save_artifacts=False, return_recommendations=False)
    raw_model = raw_artifacts["model"].value
    raw_idx2item = raw_artifacts["idx2item"].value
    item_codes.update(str(item_code) for item_code in raw_idx2item.values())
    fallback_users = set(int(user_id) for user_id in raw_model.customer_lookup_) - covered_customers
    fallback_exported = 0
    with row_jsonl.open("a", encoding="utf-8") as handle:
        for customer_id in sorted(fallback_users):
            scores = row_scores_for_customer(
                model=raw_model,
                idx2item=raw_idx2item,
                customer_id=int(customer_id),
                top_k=int(args.top_k_per_row),
            )
            item_codes.update(scores)
            handle.write(json.dumps({"CustomerID": int(customer_id), "scores": scores}) + "\n")
            fallback_exported += 1
    print(f"[final-export] raw fallback: exported_rows={fallback_exported}")

    print(f"[final-export] writing wide CSV with {len(item_codes)} item columns: {output_csv}")
    write_wide_csv(row_jsonl, output_csv, item_codes)

    manifest = {
        "output_csv": str(output_csv),
        "runtime_root": str(runtime_root),
        "output_root": str(output_root),
        "row_jsonl": str(row_jsonl),
        "item_column_count": len(item_codes),
        "covered_cohort_customer_count": len(covered_customers),
        "fallback_customer_count": fallback_exported,
        "hyperparameters": {
            "n_neighbors": int(args.n_neighbors),
            "customer_customer_weight": float(args.customer_customer_weight),
            "customer_item_weight": float(args.customer_item_weight),
            "beta": float(args.beta),
            "katz_max_nnz_per_row": int(args.katz_max_nnz_per_row),
            "top_k_per_row": int(args.top_k_per_row),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[final-export] done")


if __name__ == "__main__":
    main()
