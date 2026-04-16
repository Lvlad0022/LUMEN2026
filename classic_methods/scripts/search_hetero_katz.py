"""Grid search runner for the heterogeneous customer-item-group-family Katz pipeline."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSIC_ROOT = REPO_ROOT / "classic_methods"
SRC_DIR = CLASSIC_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation import ValidationPipeline


SEARCH_SPACE = {
    "customer_item_weight": [0.5, 1.0, 2.0],
    "item_group_weight": [0.25, 0.5],
    "item_family_weight": [0.25, 0.5],
    "group_family_weight": [ 0.25, 0.5],
    "beta": [0.5, 0.2, 0.05, 0.01],
}


@dataclass(frozen=True)
class SearchPaths:
    """Container for runtime config and output paths for one search run."""

    run_id: str
    config_dir: Path
    output_dir: Path
    base_config_path: Path
    validation_config_path: Path
    pipeline_config_path: Path
    graph_config_path: Path
    katz_config_path: Path
    summary_path: Path
    config_artifact_path: Path


def parse_args() -> argparse.Namespace:
    # Inputs: command-line invocation. Outputs: parsed search runner arguments.
    parser = argparse.ArgumentParser(description="Run a grid search over heterogeneous Katz hyperparameters.")
    parser.add_argument(
        "--search-name",
        default="hetero_katz_search",
        help="Folder name under config/search_runtime and output_search used for this search.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Validation repetitions per hyperparameter combination.",
    )
    parser.add_argument(
        "--users-per-repetition",
        type=int,
        default=20,
        help="Number of sampled users per validation repetition.",
    )
    parser.add_argument(
        "--recommendation-k",
        type=int,
        default=10,
        help="Number of recommendations to generate per user.",
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        default=10,
        help="Top-k cutoff used for validation metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of parameter combinations to execute.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove any previous runtime configs and outputs for this search before starting.",
    )
    return parser.parse_args()


def _grid_rows(search_space: dict[str, list[float]]) -> list[dict[str, float]]:
    # Inputs: hyperparameter search space. Outputs: deterministic list of grid-search parameter combinations.
    names = list(search_space)
    rows: list[dict[str, float]] = []
    for values in itertools.product(*(search_space[name] for name in names)):
        rows.append({name: float(value) for name, value in zip(names, values)})
    return rows


def _prepare_search_roots(search_name: str, *, clean: bool) -> tuple[Path, Path]:
    # Inputs: search name and cleanup flag. Outputs: runtime-config root and output root paths.
    config_root = CLASSIC_ROOT / "config" / "search_runtime" / search_name
    output_root = CLASSIC_ROOT / "output_search" / search_name
    if clean:
        if config_root.exists():
            shutil.rmtree(config_root, ignore_errors=True)
        if output_root.exists():
            shutil.rmtree(output_root, ignore_errors=True)
    config_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    return config_root, output_root


def _build_run_paths(config_root: Path, output_root: Path, run_number: int) -> SearchPaths:
    # Inputs: search roots and 1-based run number. Outputs: all runtime config and output paths for one run.
    run_id = f"run_{run_number:04d}"
    config_dir = config_root / run_id
    output_dir = output_root / run_id
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SearchPaths(
        run_id=run_id,
        config_dir=config_dir,
        output_dir=output_dir,
        base_config_path=config_dir / "base.yaml",
        validation_config_path=config_dir / "validation.yaml",
        pipeline_config_path=config_dir / "pipeline.yaml",
        graph_config_path=config_dir / "from_customer_item_group_family.yaml",
        katz_config_path=config_dir / "katz.yaml",
        summary_path=output_dir / "validation_summary.json",
        config_artifact_path=output_dir / "validation_config_bundle.json",
    )


def _write_text(path: Path, text: str) -> None:
    # Inputs: destination path and file content. Outputs: UTF-8 file written to disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def _write_runtime_configs(
    paths: SearchPaths,
    params: dict[str, float],
    *,
    repetitions: int,
    users_per_repetition: int,
    recommendation_k: int,
    metrics_k: int,
) -> None:
    # Inputs: run paths, selected hyperparameters, and validation settings. Outputs: runtime YAML files for one run.
    _write_text(
        paths.graph_config_path,
        f"""
        _target_: models.similarity_matrix.from_customer_item_group_family.customer_item_group_family_to_similarity
        method: fit
        params:
          customer_col: CustomerID
          item_col: Item Code
          item_idx_col: item_idx
          group_col: Product group
          family_col: Product family
          customer_item_weight: {params['customer_item_weight']}
          item_group_weight: {params['item_group_weight']}
          item_family_weight: {params['item_family_weight']}
          group_family_weight: {params['group_family_weight']}
          drop_missing_customer_id: true
          drop_missing_item_code: true
          drop_missing_group: true
          drop_missing_family: true
        inputs:
          dataframe: dataframe
        """,
    )

    _write_text(
        paths.katz_config_path,
        f"""
        _target_: models.similarity_matrix.katz.KatzRecommender
        method: fit
        params:
          beta: {params['beta']}
          max_iter: 100
          tol: 1.0e-10
          include_self: false
          symmetric_graph_normalization: true
        inputs:
          combined_matrix: combined_matrix
          customer_idx: customer_idx
          item_idx: item_idx
        """,
    )

    _write_text(
        paths.pipeline_config_path,
        f"""
        data:
          csv_path: ${{paths.processed_data_csv}}

        output:
          dir: {paths.output_dir}

        save_intermediates: false

        stage1:
          config_path: config/recommendation/data_processing/preprocessing_impute_only.yaml

        stage2:
          config_path: {paths.graph_config_path}

        stage3:
          config_path: {paths.katz_config_path}
        """,
    )

    _write_text(
        paths.validation_config_path,
        f"""
        seed: ${{project.seed}}

        data:
          csv_path: ${{paths.processed_data_csv}}
          user_col: CustomerID
          item_col: item_idx

        pipeline:
          recommendation_config_path: {paths.pipeline_config_path}
          recommendation_k: {recommendation_k}

        sampling:
          users_per_repetition: {users_per_repetition}

        masking:
          strategy: remove_whole_item

        metrics:
          k: {metrics_k}

        output:
          summary_path: {paths.summary_path}
          config_artifact_path: {paths.config_artifact_path}

        mlflow:
          enabled: false
          tracking_uri: null
          experiment_name: hetero_katz_search
          run_name: {paths.run_id}
          tags: {{}}
        """,
    )

    _write_text(
        paths.base_config_path,
        f"""
        project:
          name: classic_methods
          seed: 42

        validation_bool: true
        validation_repetitions: {repetitions}

        paths:
          config_path: config/paths.yaml

        pipeline:
          config_path: {paths.pipeline_config_path}

        validation:
          config_path: {paths.validation_config_path}
        """,
    )


def _flatten_result(paths: SearchPaths, params: dict[str, float], summary: dict[str, Any]) -> dict[str, Any]:
    # Inputs: run paths, hyperparameters, and validation summary. Outputs: flat leaderboard row.
    metrics = summary.get("metrics", {})
    row: dict[str, Any] = {
        "run_id": paths.run_id,
        "summary_path": str(paths.summary_path),
        "config_path": str(paths.base_config_path),
        "repetitions_run": int(summary.get("repetitions_run", 0)),
    }
    row.update(params)
    for metric_name in ("ndcg_at_k", "recall_at_k", "mrr_at_k"):
        metric_stats = dict(metrics.get(metric_name, {}) or {})
        row[f"{metric_name}_mean"] = float(metric_stats.get("mean", 0.0))
        row[f"{metric_name}_std"] = float(metric_stats.get("std", 0.0))
    return row


def _write_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    # Inputs: accumulated leaderboard rows and search output root. Outputs: CSV and JSONL leaderboard files.
    if not rows:
        return
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["ndcg_at_k_mean"]),
            -float(row["recall_at_k_mean"]),
            -float(row["mrr_at_k_mean"]),
            str(row["run_id"]),
        ),
    )

    csv_path = output_root / "results.csv"
    fieldnames = list(ordered[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    jsonl_path = output_root / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, default=str) + "\n")


def main() -> None:
    # Inputs: command-line args. Outputs: executed grid search with saved runtime configs and leaderboard files.
    args = parse_args()
    config_root, output_root = _prepare_search_roots(args.search_name, clean=bool(args.clean))
    combinations = _grid_rows(SEARCH_SPACE)
    if args.limit is not None:
        combinations = combinations[: max(int(args.limit), 0)]

    if not combinations:
        print("No parameter combinations to run.")
        return

    results: list[dict[str, Any]] = []
    total = len(combinations)
    for run_number, params in enumerate(combinations, start=1):
        paths = _build_run_paths(config_root, output_root, run_number)
        _write_runtime_configs(
            paths,
            params,
            repetitions=int(args.repetitions),
            users_per_repetition=int(args.users_per_repetition),
            recommendation_k=int(args.recommendation_k),
            metrics_k=int(args.metrics_k),
        )
        print(f"[search] {run_number}/{total} {paths.run_id} params={params}")
        summary = ValidationPipeline(paths.base_config_path, show_progress=True).run()
        row = _flatten_result(paths, params, summary)
        results.append(row)
        _write_results(results, output_root)
        print(
            "[search] completed "
            f"{paths.run_id} ndcg={row['ndcg_at_k_mean']:.4f} "
            f"recall={row['recall_at_k_mean']:.4f} mrr={row['mrr_at_k_mean']:.4f}"
        )

    print(f"[search] results saved to {output_root}")
    print(f"[search] leaderboard csv: {output_root / 'results.csv'}")


if __name__ == "__main__":
    main()
