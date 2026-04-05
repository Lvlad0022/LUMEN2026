"""Offline validation orchestration around the recommendation pipeline."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from .offline import mrr_at_k, ndcg_at_k, recall_at_k, remove_whole_item
    from ..pipeline import Pipeline, _deep_merge, _load_simple_yaml, _resolve_placeholders
except ImportError:  # pragma: no cover - supports direct imports from src/
    from validation.offline import mrr_at_k, ndcg_at_k, recall_at_k, remove_whole_item  # type: ignore
    from pipeline import Pipeline, _deep_merge, _load_simple_yaml, _resolve_placeholders  # type: ignore


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
    summary_path: Path
    user_col: str = "CustomerID"
    item_col: str = "item_idx"
    masking_strategy: str = "remove_whole_item"


class ValidationPipeline:
    """Loop over masked recommendation runs and aggregate evaluation metrics."""

    def __init__(self, base_config_path: str | Path) -> None:
        self.base_config_path = Path(base_config_path)
        self.project_dir = self._project_dir_from_config_path(self.base_config_path)
        self.base_config = self._load_config(self.base_config_path)
        self.paths_config = self._load_config_if_ref(self.base_config.get("paths"))
        self.validation_config_path = self._resolve_config_ref_path(self.base_config.get("validation"))
        self.validation_project_dir = (
            self._project_dir_from_config_path(self.validation_config_path)
            if self.validation_config_path is not None
            else self.project_dir
        )
        self.validation_root = self._load_config_if_ref(self.base_config.get("validation"))
        self.config_context = _deep_merge(_deep_merge(self.base_config, self.paths_config), self.validation_root)
        self.base_config = _resolve_placeholders(self.base_config, self.config_context)
        self.paths_config = _resolve_placeholders(self.paths_config, self.config_context)
        self.validation_root = _resolve_placeholders(self.validation_root, self.config_context)
        self.config_context = _deep_merge(_deep_merge(self.base_config, self.paths_config), self.validation_root)
        self.config = self._resolve_validation_config()
        runtime_parent = self.validation_project_dir / "config"
        runtime_parent.mkdir(parents=True, exist_ok=True)
        self._runtime_dir = tempfile.TemporaryDirectory(dir=runtime_parent)
        self.recommendation_base_config_path = self._build_recommendation_base_config()

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        loaded = _load_simple_yaml(path)
        if not isinstance(loaded, dict):
            raise ValueError("Validation config must be a mapping at the top level.")
        return loaded

    @staticmethod
    def _project_dir_from_config_path(config_path: Path) -> Path:
        for parent in config_path.parents:
            if parent.name == "config":
                return parent.parent
        return config_path.parent.parent

    def _load_config_if_ref(self, section: Any) -> dict[str, Any]:
        if not isinstance(section, Mapping):
            return {}
        path = self._resolve_config_ref_path(section)
        if path is None:
            return dict(section)
        return self._load_config(path)

    def _resolve_config_ref_path(self, section: Any) -> Path | None:
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
        path = Path(value)
        if not path.is_absolute():
            anchor = self.project_dir if base_dir is None else base_dir
            path = (anchor / path).resolve()
        return path

    def _resolve_validation_config(self) -> ValidationConfig:
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

        users_per_repetition = int(sampling_config.get("users_per_repetition", 0))
        if users_per_repetition <= 0:
            raise ValueError("validation.sampling.users_per_repetition must be positive.")

        recommendation_k = int(pipeline_config.get("recommendation_k", 10))
        metrics_k = int(metrics_config.get("k", recommendation_k))
        seed = int(self.validation_root.get("seed", self.base_config.get("project", {}).get("seed", 42)))
        masking_strategy = str(masking_config.get("strategy", "remove_whole_item"))
        if masking_strategy != "remove_whole_item":
            raise ValueError(f"Unsupported masking strategy: {masking_strategy}")

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
            summary_path=self._resolve_path(str(summary_path)),
            user_col=str(data_config.get("user_col", "CustomerID")),
            item_col=str(data_config.get("item_col", "item_idx")),
            masking_strategy=masking_strategy,
        )

    def run(self) -> dict[str, Any]:
        raw_df = Pipeline(self.recommendation_base_config_path)._read_dataframe(self.config.source_csv_path)
        if self.config.user_col not in raw_df.columns:
            raise KeyError(f"Missing validation user column: {self.config.user_col}")
        if self.config.item_col not in raw_df.columns:
            raise KeyError(f"Missing validation item column: {self.config.item_col}")

        unique_users = pd.unique(raw_df[self.config.user_col].dropna())
        if unique_users.size == 0:
            raise ValueError("Validation source dataframe contains no users.")

        rng = np.random.default_rng(self.config.seed)
        repetition_results: list[dict[str, Any]] = []
        user_test_counts: dict[int, int] = {}

        for repetition_number in range(1, self.config.repetitions + 1):
            sampled_users = self._sample_users(unique_users, rng)
            split = remove_whole_item(
                raw_df,
                sampled_users.tolist(),
                user_col=self.config.user_col,
                item_col=self.config.item_col,
                rng=rng,
            )

            recommendation_pipeline = Pipeline(self.recommendation_base_config_path)
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

        summary = self._build_summary(repetition_results, user_test_counts)
        self._save_summary(summary)
        return summary

    def _sample_users(self, unique_users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sample_size = min(self.config.users_per_repetition, int(unique_users.shape[0]))
        selected_positions = rng.choice(unique_users.shape[0], size=sample_size, replace=False)
        return unique_users[selected_positions]

    def _build_summary(
        self,
        repetition_results: list[dict[str, Any]],
        user_test_counts: dict[int, int],
    ) -> dict[str, Any]:
        metric_names = ("ndcg_at_k", "recall_at_k", "mrr_at_k")
        metrics_summary: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            values = np.asarray([float(result[metric_name]) for result in repetition_results], dtype=float)
            metrics_summary[metric_name] = {
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std(ddof=0)) if values.size else 0.0,
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
            "per_user_test_counts": {str(user_id): count for user_id, count in sorted(user_test_counts.items())},
            "repetition_results": repetition_results,
        }

    def _save_summary(self, summary: Mapping[str, Any]) -> None:
        self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _build_recommendation_base_config(self) -> Path:
        runtime_base_path = Path(self._runtime_dir.name) / "recommendation_base.yaml"
        project_config = dict(self.base_config.get("project", {}) or {})
        project_name = project_config.get("name", "classic_methods")
        project_seed = project_config.get("seed", 42)

        content = (
            "project:\n"
            f"  name: {project_name}\n"
            f"  seed: {project_seed}\n\n"
            "paths:\n"
            f"  config_path: {self._resolve_path(str(self.base_config['paths']['config_path']))}\n\n"
            "pipeline:\n"
            f"  config_path: {self.config.recommendation_pipeline_config_path}\n"
        )
        runtime_base_path.write_text(content, encoding="utf-8")
        return runtime_base_path


__all__ = ["ValidationPipeline"]
