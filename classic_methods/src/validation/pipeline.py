"""Offline validation orchestration around the recommendation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

try:
    from .offline import (
        EvaluationSplitResult,
        hold_out_last_novel_tail_item,
        mrr_at_k,
        ndcg_at_k,
        recall_at_k,
        remove_whole_item,
    )
    from ..pipeline import Pipeline, _deep_merge, _load_simple_yaml, _resolve_placeholders
    from ..utils import ResolvedValidationConfig, resolve_validation_config
except ImportError:  # pragma: no cover - supports direct imports from src/
    from validation.offline import (  # type: ignore
        EvaluationSplitResult,
        hold_out_last_novel_tail_item,
        mrr_at_k,
        ndcg_at_k,
        recall_at_k,
        remove_whole_item,
    )
    from pipeline import Pipeline, _deep_merge, _load_simple_yaml, _resolve_placeholders  # type: ignore
    from utils import ResolvedValidationConfig, resolve_validation_config  # type: ignore


@dataclass(frozen=True)
class ValidationConfig:
    """Resolved validation runtime configuration."""

    repetitions: int
    seed: int
    users_per_repetition: int
    recommendation_k: int
    metrics_k: int
    source_csv_path: Path
    recommendation_pipeline_config_path: Path
    output_dir: Path
    summary_path: Path
    config_artifact_path: Path
    mlflow_dir: Path
    user_col: str = "CustomerID"
    item_col: str = "item_idx"
    date_col: str = "Order Date"
    masking_strategy: str = "remove_whole_item"
    tail_fraction: float = 0.2
    min_training_customer_support: int = 0
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_tags: dict[str, str] | None = None


def _iter_validation_progress(iterable: list[int], *, enabled: bool, total: int) -> Any:
    # Inputs: repetition numbers, progress flag, and total count. Outputs: repetition iterator with optional tqdm.
    if enabled and tqdm is not None:
        return tqdm(iterable, desc="Validation repetitions", total=total, unit="rep")
    return iterable


def _json_safe(value: Any) -> Any:
    # Inputs: arbitrary nested value. Outputs: equivalent structure using JSON-serializable Python primitives.
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class ValidationPipeline:
    """Loop over masked recommendation runs and aggregate evaluation metrics."""

    def __init__(self, base_config_path: str | Path, *, show_progress: bool = True) -> None:
        # Inputs: base config path and progress flag. Outputs: initialized validation pipeline.
        self.base_config_path = Path(base_config_path)
        self.show_progress = bool(show_progress)
        self.resolved = resolve_validation_config(self.base_config_path)
        self.project_dir = self.resolved.project_dir
        self.base_config = self.resolved.root_config
        self.paths_config = self.resolved.paths_config
        self.validation_root = self.resolved.validation_config
        self.validation_config_path = self._resolve_config_ref_path(self.base_config.get("validation"))
        self.validation_project_dir = self.project_dir
        self.config_context = self.resolved.merged_validation_config
        self.config = self._resolve_validation_config()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        # Inputs: config path. Outputs: parsed validation config mapping.
        loaded = _load_simple_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError("Validation config must be a mapping at the top level.")
        return loaded

    @staticmethod
    def _project_dir_from_config_path(config_path: Path) -> Path:
        # Inputs: config path. Outputs: inferred project directory.
        for parent in config_path.parents:
            if parent.name == "config":
                return parent.parent
        return config_path.parent.parent

    def _load_config_if_ref(self, section: Any) -> dict[str, Any]:
        # Inputs: config section. Outputs: inline mapping or loaded referenced config.
        if not isinstance(section, Mapping):
            return {}
        path = self._resolve_config_ref_path(section)
        if path is None:
            return dict(section)
        return self._load_config(path)

    def _resolve_config_ref_path(self, section: Any) -> Path | None:
        # Inputs: config section. Outputs: resolved referenced config path when present.
        if not isinstance(section, Mapping):
            return None
        config_path = section.get("config_path")
        if not config_path:
            return None
        path = Path(str(config_path))
        if not path.is_absolute():
            path = (self.project_dir / path).resolve()
        return path

    def _resolve_path(self, value: str, *, base_dir: Path | None = None) -> Path:
        # Inputs: path string and optional base dir. Outputs: resolved absolute path.
        path = Path(value)
        if not path.is_absolute():
            anchor = self.project_dir if base_dir is None else base_dir
            path = (anchor / path).resolve()
        return path

    def _next_validation_run_dir(self, validation_root: Path) -> Path:
        # Inputs: dated validation output root. Outputs: next numbered validation run directory.
        validation_root.mkdir(parents=True, exist_ok=True)
        run_numbers: list[int] = []
        for child in validation_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("run"):
                continue
            suffix = name[3:]
            if suffix.isdigit():
                run_numbers.append(int(suffix))
        next_run = max(run_numbers, default=0) + 1
        return validation_root / f"run{next_run}"

    def _resolve_validation_output_paths(
        self,
        output_config: Mapping[str, Any],
        *,
        output_root: Path,
        mlflow_dir: Path,
    ) -> tuple[Path, Path, Path]:
        # Inputs: output config plus canonical roots. Outputs: validation run dir, summary path, and config-bundle path.
        configured_summary = output_config.get("summary_path")
        configured_bundle = output_config.get("config_artifact_path")

        legacy_summary = (output_root / "validation_summary.json").resolve()
        legacy_bundle = (mlflow_dir / "validation_config_bundle.json").resolve()

        resolved_summary = (
            self._resolve_path(str(configured_summary))
            if configured_summary
            else legacy_summary
        )
        resolved_bundle = (
            self._resolve_path(str(configured_bundle))
            if configured_bundle
            else legacy_bundle
        )

        uses_legacy_layout = resolved_summary == legacy_summary and resolved_bundle == legacy_bundle
        if uses_legacy_layout:
            today = datetime.now().date().isoformat()
            dated_root = output_root / "validation" / today
            run_dir = self._next_validation_run_dir(dated_root)
            return (
                run_dir,
                run_dir / "validation_summary.json",
                run_dir / "validation_config_bundle.json",
            )

        run_dir = resolved_summary.parent
        return run_dir, resolved_summary, resolved_bundle

    def _resolve_validation_config(self) -> ValidationConfig:
        # Inputs: loaded config tree. Outputs: normalized runtime validation config.
        repetitions = int(self.base_config.get("validation_repetitions", 1))
        if repetitions <= 0:
            raise ValueError("validation_repetitions must be positive.")

        data_config = dict(self.validation_root.get("data", {}) or {})
        sampling_config = dict(self.validation_root.get("sampling", {}) or {})
        masking_config = dict(self.validation_root.get("masking", {}) or {})
        metrics_config = dict(self.validation_root.get("metrics", {}) or {})
        output_config = dict(self.validation_root.get("output", {}) or {})
        pipeline_config = dict(self.validation_root.get("pipeline", {}) or {})

        source_csv_path = data_config.get("csv_path")
        if not source_csv_path:
            raise ValueError("validation.data.csv_path is required.")
        recommendation_pipeline_config_path = pipeline_config.get("recommendation_config_path")
        if not recommendation_pipeline_config_path:
            raise ValueError("validation.pipeline.recommendation_config_path is required.")
        summary_path = output_config.get("summary_path")
        if not summary_path:
            raise ValueError("validation.output.summary_path is required.")
        paths_root = dict(self.paths_config.get("paths", {}) or {})
        output_root = paths_root.get("output_dir")
        if not output_root:
            raise ValueError("paths.output_dir is required for validation outputs.")
        mlflow_dir = paths_root.get("mlflow_dir")
        if not mlflow_dir:
            raise ValueError("paths.mlflow_dir is required for validation logging/config artifacts.")
        resolved_output_root = self._resolve_path(str(output_root))
        resolved_mlflow_dir = self._resolve_path(str(mlflow_dir))
        validation_output_dir, resolved_summary_path, resolved_config_artifact_path = self._resolve_validation_output_paths(
            output_config,
            output_root=resolved_output_root,
            mlflow_dir=resolved_mlflow_dir,
        )

        users_per_repetition = int(sampling_config.get("users_per_repetition", 0))
        if users_per_repetition <= 0:
            raise ValueError("validation.sampling.users_per_repetition must be positive.")

        recommendation_k = int(pipeline_config.get("recommendation_k", 10))
        metrics_k = int(metrics_config.get("k", recommendation_k))
        seed = int(self.validation_root.get("seed", self.base_config.get("project", {}).get("seed", 42)))
        masking_strategy = str(masking_config.get("strategy", "remove_whole_item"))
        if masking_strategy not in {"remove_whole_item", "last_novel_tail_item"}:
            raise ValueError(f"Unsupported masking strategy: {masking_strategy}")
        tail_fraction = float(masking_config.get("tail_fraction", 0.2))
        min_training_customer_support = int(masking_config.get("min_training_customer_support", 0))

        mlflow_config = dict(self.validation_root.get("mlflow", {}) or {})
        mlflow_tags = {
            str(key): str(value)
            for key, value in dict(mlflow_config.get("tags", {}) or {}).items()
        } or None

        return ValidationConfig(
            repetitions=repetitions,
            seed=seed,
            users_per_repetition=users_per_repetition,
            recommendation_k=recommendation_k,
            metrics_k=metrics_k,
            source_csv_path=self._resolve_path(str(source_csv_path)),
            recommendation_pipeline_config_path=self._resolve_path(
                str(recommendation_pipeline_config_path),
                base_dir=self.validation_project_dir,
            ),
            output_dir=validation_output_dir,
            summary_path=resolved_summary_path,
            config_artifact_path=resolved_config_artifact_path,
            mlflow_dir=resolved_mlflow_dir,
            user_col=str(data_config.get("user_col", "CustomerID")),
            item_col=str(data_config.get("item_col", "item_idx")),
            date_col=str(data_config.get("date_col", "Order Date")),
            masking_strategy=masking_strategy,
            tail_fraction=tail_fraction,
            min_training_customer_support=min_training_customer_support,
            mlflow_enabled=bool(mlflow_config.get("enabled", False)),
            mlflow_tracking_uri=(
                str(mlflow_config.get("tracking_uri"))
                if mlflow_config.get("tracking_uri") is not None
                else None
            ),
            mlflow_experiment_name=(
                str(mlflow_config.get("experiment_name"))
                if mlflow_config.get("experiment_name") is not None
                else None
            ),
            mlflow_run_name=(
                str(mlflow_config.get("run_name"))
                if mlflow_config.get("run_name") is not None
                else None
            ),
            mlflow_tags=mlflow_tags,
        )

    def run(self) -> dict[str, Any]:
        # Inputs: none. Outputs: saved validation summary and metrics payload.
        raw_df = self._build_recommendation_pipeline()._read_dataframe(self.config.source_csv_path)
        if self.config.user_col not in raw_df.columns:
            raise KeyError(f"Missing validation user column: {self.config.user_col}")
        if self.config.item_col not in raw_df.columns:
            raise KeyError(f"Missing validation item column: {self.config.item_col}")
        if self.config.masking_strategy == "last_novel_tail_item" and self.config.date_col not in raw_df.columns:
            raise KeyError(f"Missing validation date column: {self.config.date_col}")

        unique_users = pd.unique(raw_df[self.config.user_col].dropna())
        if unique_users.size == 0:
            raise ValueError("Validation source dataframe contains no users.")

        rng = np.random.default_rng(self.config.seed)
        repetition_results: list[dict[str, Any]] = []
        user_test_counts: dict[int, int] = {}
        user_item_counts = (
            raw_df.groupby(self.config.user_col)[self.config.item_col]
            .nunique(dropna=True)
            .to_dict()
        )
        item_user_counts = (
            raw_df.groupby(self.config.item_col)[self.config.user_col]
            .nunique(dropna=True)
            .to_dict()
        )
        case_diagnostics = {
            "correct": {
                "user_num_items_bought": [],
                "item_num_users_bought": [],
            },
            "incorrect": {
                "user_num_items_bought": [],
                "item_num_users_bought": [],
            },
        }
        split_metadata: dict[str, Any] | None = None
        fixed_split: EvaluationSplitResult | None = None
        fixed_eval_users: np.ndarray | None = None

        if self.config.masking_strategy == "last_novel_tail_item":
            fixed_split = hold_out_last_novel_tail_item(
                raw_df,
                user_col=self.config.user_col,
                item_col=self.config.item_col,
                date_col=self.config.date_col,
                tail_fraction=self.config.tail_fraction,
                min_training_customer_support=self.config.min_training_customer_support,
            )
            fixed_eval_users = np.asarray(fixed_split.evaluated_user_ids)
            split_metadata = _json_safe(dict(fixed_split.split_details or {}))
            if self.show_progress:
                print(
                    "[validation] novel-tail split: "
                    f"validation_customers={len(fixed_split.evaluated_user_ids)}, "
                    f"training_purchases={len(fixed_split.masked_df)}, "
                    f"removed_purchases={len(raw_df) - len(fixed_split.masked_df)}"
                )
            if fixed_eval_users.size == 0:
                raise ValueError("Novel-tail validation produced zero eligible validation customers.")

        if self.show_progress:
            if self.config.masking_strategy == "last_novel_tail_item":
                print("[validation] starting deterministic novel-tail validation")
            else:
                print(
                    f"[validation] starting {self.config.repetitions} repetition(s), "
                    f"{self.config.users_per_repetition} sampled user(s) each"
                )

        repetition_count = 1 if self.config.masking_strategy == "last_novel_tail_item" else self.config.repetitions
        repetition_numbers = list(range(1, repetition_count + 1))
        repetition_iterable = _iter_validation_progress(
            repetition_numbers,
            enabled=self.show_progress,
            total=repetition_count,
        )

        for repetition_number in repetition_iterable:
            if self.show_progress:
                if self.config.masking_strategy == "last_novel_tail_item":
                    print("[validation] deterministic split: using all eligible validation customers")
                else:
                    print(f"[validation] repetition {repetition_number}/{self.config.repetitions}: sampling users")
            if self.config.masking_strategy == "last_novel_tail_item":
                assert fixed_split is not None
                assert fixed_eval_users is not None
                sampled_users = fixed_eval_users
                sampled_user_set = set(sampled_users.tolist())
                split = EvaluationSplitResult(
                    masked_df=fixed_split.masked_df,
                    ground_truth={
                        user_id: item_id
                        for user_id, item_id in fixed_split.ground_truth.items()
                        if user_id in sampled_user_set
                    },
                    evaluated_user_ids=[user_id for user_id in fixed_split.evaluated_user_ids if user_id in sampled_user_set],
                    skipped_user_ids=[user_id for user_id in fixed_split.evaluated_user_ids if user_id not in sampled_user_set],
                    split_details=fixed_split.split_details,
                )
            else:
                sampled_users = self._sample_users(unique_users, rng)
                split = remove_whole_item(
                    raw_df,
                    sampled_users.tolist(),
                    user_col=self.config.user_col,
                    item_col=self.config.item_col,
                    rng=rng,
                )

            if self.show_progress:
                if self.config.masking_strategy == "last_novel_tail_item":
                    print(
                        "[validation] deterministic split: "
                        f"evaluating {len(split.evaluated_user_ids)} user(s), "
                        f"training rows={len(split.masked_df)}"
                    )
                else:
                    print(
                        f"[validation] repetition {repetition_number}/{self.config.repetitions}: "
                        f"masked {len(split.evaluated_user_ids)} user(s), skipped {len(split.skipped_user_ids)}"
                    )

            recommendation_pipeline = self._build_recommendation_pipeline()
            artifacts = recommendation_pipeline.run(
                initial_artifacts={"dataframe": split.masked_df},
                save_artifacts=False,
                return_recommendations=True,
                recommendation_k=self.config.recommendation_k,
            )
            predictions = artifacts["recommendations"].value

            evaluated_ground_truth = {
                int(user_id): split.ground_truth[user_id]
                for user_id in split.evaluated_user_ids
                if int(user_id) in predictions
            }
            users_missing_predictions = [
                int(user_id) for user_id in split.evaluated_user_ids if int(user_id) not in predictions
            ]

            for user_id in evaluated_ground_truth:
                user_test_counts[user_id] = user_test_counts.get(user_id, 0) + 1
                target_item = evaluated_ground_truth[user_id]
                top_items = predictions.get(int(user_id), [])[: self.config.metrics_k]
                bucket = "correct" if target_item in top_items else "incorrect"
                case_diagnostics[bucket]["user_num_items_bought"].append(
                    int(user_item_counts.get(user_id, 0))
                )
                case_diagnostics[bucket]["item_num_users_bought"].append(
                    int(item_user_counts.get(target_item, 0))
                )

            repetition_result = {
                "repetition": repetition_number,
                "sampled_user_ids": [int(user_id) for user_id in sampled_users.tolist()],
                "sampled_user_count": int(len(sampled_users)),
                "masked_user_count": int(len(split.evaluated_user_ids)),
                "evaluated_user_count": int(len(evaluated_ground_truth)),
                "skipped_user_ids": [int(user_id) for user_id in split.skipped_user_ids],
                "users_missing_predictions": users_missing_predictions,
                "ndcg_at_k": ndcg_at_k(predictions, evaluated_ground_truth, k=self.config.metrics_k),
                "recall_at_k": recall_at_k(predictions, evaluated_ground_truth, k=self.config.metrics_k),
                "mrr_at_k": mrr_at_k(predictions, evaluated_ground_truth, k=self.config.metrics_k),
            }
            repetition_results.append(repetition_result)
            if self.show_progress:
                if self.config.masking_strategy == "last_novel_tail_item":
                    print(
                        "[validation] deterministic result: "
                        f"ndcg={repetition_result['ndcg_at_k']:.4f}, "
                        f"recall={repetition_result['recall_at_k']:.4f}, "
                        f"mrr={repetition_result['mrr_at_k']:.4f}"
                    )
                else:
                    print(
                        f"[validation] repetition {repetition_number}/{self.config.repetitions}: "
                        f"ndcg={repetition_result['ndcg_at_k']:.4f}, "
                        f"recall={repetition_result['recall_at_k']:.4f}, "
                        f"mrr={repetition_result['mrr_at_k']:.4f}"
                    )

        summary = self._build_summary(repetition_results, user_test_counts, case_diagnostics)
        if split_metadata is not None:
            summary["split_details"] = split_metadata
        self._save_summary(summary)
        self._save_config_artifact(summary)
        self._log_mlflow(summary)
        if self.show_progress:
            print(f"[validation] complete, summary written to {self.config.summary_path}")
        return summary

    def _sample_users(self, unique_users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Inputs: unique users and RNG. Outputs: sampled user ids for one repetition.
        sample_size = min(self.config.users_per_repetition, int(unique_users.shape[0]))
        selected_positions = rng.choice(unique_users.shape[0], size=sample_size, replace=False)
        return unique_users[selected_positions]

    def _build_summary(
        self,
        repetition_results: list[dict[str, Any]],
        user_test_counts: dict[int, int],
        case_diagnostics: dict[str, dict[str, list[int]]],
    ) -> dict[str, Any]:
        # Inputs: per-repetition metrics and user counts. Outputs: aggregate validation summary.
        metric_names = ("ndcg_at_k", "recall_at_k", "mrr_at_k")
        metrics_summary: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            values = np.asarray([float(result[metric_name]) for result in repetition_results], dtype=float)
            metrics_summary[metric_name] = {
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std(ddof=0)) if values.size else 0.0,
            }

        prediction_case_summary = {
            bucket: {
                "count": int(len(stats["user_num_items_bought"])),
                "user_num_items_bought": self._mean_median(stats["user_num_items_bought"]),
                "item_num_users_bought": self._mean_median(stats["item_num_users_bought"]),
            }
            for bucket, stats in case_diagnostics.items()
        }

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "requested_repetitions": self.config.repetitions,
            "repetitions_run": len(repetition_results),
            "seed": self.config.seed,
            "source_csv_path": str(self.config.source_csv_path),
            "masking_strategy": self.config.masking_strategy,
            "users_per_repetition": self.config.users_per_repetition,
            "recommendation_k": self.config.recommendation_k,
            "metrics_k": self.config.metrics_k,
            "metrics": metrics_summary,
            "prediction_case_diagnostics": prediction_case_summary,
            "per_user_test_counts": {str(user_id): count for user_id, count in sorted(user_test_counts.items())},
            "repetition_results": repetition_results,
        }

    @staticmethod
    def _mean_median(values: list[int]) -> dict[str, float]:
        # Inputs: integer value list. Outputs: mean and median summary for that list.
        if not values:
            return {"mean": 0.0, "median": 0.0}
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
        }

    def _save_summary(self, summary: Mapping[str, Any]) -> None:
        # Inputs: summary payload. Outputs: summary JSON written to disk.
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _save_config_artifact(self, summary: Mapping[str, Any]) -> None:
        # Inputs: validation summary. Outputs: validation config bundle written to disk.
        """Persist one validation config bundle next to the final summary artifact."""

        self.config.config_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_config_path": str(self.base_config_path),
            "validation_config_path": str(self.validation_config_path) if self.validation_config_path is not None else None,
            "recommendation_pipeline_config_path": str(self.config.recommendation_pipeline_config_path),
            "base_config": self.base_config,
            "paths_config": self.paths_config,
            "validation_root": self.validation_root,
            "resolved_recommendation_pipeline": self.resolved.recommendation.merged_config,
            "resolved_validation_config": {
                "repetitions": self.config.repetitions,
                "seed": self.config.seed,
                "users_per_repetition": self.config.users_per_repetition,
                "recommendation_k": self.config.recommendation_k,
                "metrics_k": self.config.metrics_k,
                "source_csv_path": str(self.config.source_csv_path),
                "recommendation_pipeline_config_path": str(self.config.recommendation_pipeline_config_path),
                "summary_path": str(self.config.summary_path),
                "config_artifact_path": str(self.config.config_artifact_path),
                "mlflow_dir": str(self.config.mlflow_dir),
                "user_col": self.config.user_col,
                "item_col": self.config.item_col,
                "date_col": self.config.date_col,
                "masking_strategy": self.config.masking_strategy,
                "tail_fraction": self.config.tail_fraction,
                "min_training_customer_support": self.config.min_training_customer_support,
                "mlflow_enabled": self.config.mlflow_enabled,
                "mlflow_tracking_uri": self.config.mlflow_tracking_uri,
                "mlflow_experiment_name": self.config.mlflow_experiment_name,
                "mlflow_run_name": self.config.mlflow_run_name,
                "mlflow_tags": self.config.mlflow_tags,
            },
            "summary_excerpt": {
                "repetitions_run": summary.get("repetitions_run"),
                "metrics": summary.get("metrics"),
            },
        }
        self.config.config_artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _log_mlflow(self, summary: Mapping[str, Any]) -> None:
        # Inputs: validation summary. Outputs: metrics and artifacts logged to MLflow when enabled.
        """Log validation config, metrics, and tags to MLflow when validation logging is enabled."""

        if not self.config.mlflow_enabled:
            return
        if mlflow is None:
            raise ImportError("mlflow logging is enabled in validation config, but mlflow is not installed.")

        tracking_uri = self.config.mlflow_tracking_uri
        if tracking_uri is None:
            tracking_uri = self.config.mlflow_dir.resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        if self.config.mlflow_experiment_name:
            mlflow.set_experiment(self.config.mlflow_experiment_name)

        tags = dict(self.config.mlflow_tags or {})
        tags.setdefault("pipeline_mode", "validation")
        tags.setdefault("masking_strategy", self.config.masking_strategy)
        tags.setdefault("recommendation_config_path", str(self.config.recommendation_pipeline_config_path))

        with mlflow.start_run(run_name=self.config.mlflow_run_name):
            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    "repetitions": self.config.repetitions,
                    "seed": self.config.seed,
                    "users_per_repetition": self.config.users_per_repetition,
                    "recommendation_k": self.config.recommendation_k,
                    "metrics_k": self.config.metrics_k,
                    "source_csv_path": str(self.config.source_csv_path),
                    "recommendation_pipeline_config_path": str(self.config.recommendation_pipeline_config_path),
                    "masking_strategy": self.config.masking_strategy,
                    "tail_fraction": self.config.tail_fraction,
                    "min_training_customer_support": self.config.min_training_customer_support,
                }
            )
            metrics = summary.get("metrics", {})
            for metric_name, metric_stats in metrics.items():
                if not isinstance(metric_stats, Mapping):
                    continue
                for stat_name, metric_value in metric_stats.items():
                    mlflow.log_metric(f"{metric_name}_{stat_name}", float(metric_value))
            mlflow.log_artifact(str(self.config.config_artifact_path))
            mlflow.log_artifact(str(self.config.summary_path))

    def _build_recommendation_pipeline(self) -> Pipeline:
        # Inputs: resolved recommendation config bundle. Outputs: configured Pipeline instance.
        """Build the recommendation pipeline directly from the resolved merged config bundle."""

        recommendation = self.resolved.recommendation
        return Pipeline(
            recommendation.base_config_path,
            root_config=recommendation.root_config,
            paths_config=recommendation.paths_config,
            pipeline_config=recommendation.pipeline_config,
            project_dir=recommendation.project_dir,
            show_progress=self.show_progress,
        )


__all__ = ["ValidationPipeline"]
